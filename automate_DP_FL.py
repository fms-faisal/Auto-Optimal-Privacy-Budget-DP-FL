import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader  # Added DPDataLoader import
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
import flwr as fl
from models import DenseNet121
from typing import Dict, List, Tuple, Optional
from flwr.common import Metrics, NDArrays
from flwr.common import parameters_to_ndarrays
from flwr.common import Parameters

# Privacy hyperparameters
MAX_GRAD_NORM = 1.2
DELTA = 1e-5
BATCH_SIZE = 16
MAX_PHYSICAL_BATCH_SIZE = 16
EPOCHS = 2

# Paths
train_csv_path = "train.csv"
train_img_dir = "train_images"
test_csv_path = "test.csv"
test_img_dir = "test_images"
save_model_path = "densenet_dynamic_epsilon_final.pth"
log_file = "training_log_final.txt"

# Federated learning parameters
NUM_CLIENTS = 3
NUM_ROUNDS = 3
CANDIDATE_EPSILONS = [0.5, 2.0, 3.0, 3.5, 4.0, 7.0, 10.0, 20.0]
w1, w2 = 0.5, 0.5
proxy_data_ratio = .5


class MultiLabelEncoder:
    def __init__(self):
        self.label_encoders = {}

    def fit_transform(self, df, columns):
        for col in columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        return df

    def transform(self, df, columns):
        for col in columns:
            le = self.label_encoders[col]
            df[col] = le.transform(df[col])
        return df


class SkinLesionDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.image_extensions = [".jpeg", ".jpg", ".png"]

    def __len__(self):
        return len(self.df)

    def find_image_path(self, image_id):
        for ext in self.image_extensions:
            img_path = os.path.join(self.img_dir, f"{image_id}{ext}")
            if os.path.exists(img_path):
                return img_path
        raise FileNotFoundError(f"Image {image_id} not found in {self.img_dir} with extensions {self.image_extensions}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        img_path = self.find_image_path(image_id)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = {
            "super_class": torch.tensor(row["super_class"], dtype=torch.long),
            "malignancy": torch.tensor(row["malignancy"], dtype=torch.long),
            "main_class_1": torch.tensor(row["main_class_1"], dtype=torch.long),
            "main_class_2": torch.tensor(row["main_class_2"], dtype=torch.long),
            "sub_class": torch.tensor(row["sub_class"], dtype=torch.long),
        }
        return image, labels


def log_message(message, file_path=log_file, mode="a"):
    print(message)
    with open(file_path, mode) as f:
        f.write(message + "\n")


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, train_loader, val_loader, device):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.current_epsilon = CANDIDATE_EPSILONS[0]

        self.model = DenseNet121().to(device)
        self.model = ModuleValidator.fix(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=8.672307011698221e-05)
        self.privacy_engine = None
        self.criterion_dict = {
            task: nn.CrossEntropyLoss().to(device)
            for task in ["super_class", "malignancy", "main_class_1", "main_class_2", "sub_class"]
        }

    def _init_privacy_engine(self, epsilon):
        current_state = self.model.state_dict()
        self.model = DenseNet121().to(self.device)
        self.model = ModuleValidator.fix(self.model)
        self.model.load_state_dict(current_state)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=8.672307011698221e-05)

        self.privacy_engine = PrivacyEngine()
        (self.model, self.optimizer, self.train_loader) = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            epochs=EPOCHS,
            target_epsilon=epsilon,
            target_delta=DELTA,
            max_grad_norm=MAX_GRAD_NORM,
        )

    def get_parameters(self, config) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, str]) -> Tuple[NDArrays, int, Dict]:
        self.set_parameters(parameters)
        new_epsilon = float(config.get("epsilon", CANDIDATE_EPSILONS[0]))
        self.current_epsilon = new_epsilon  # FIXED: Update current epsilon from server config
        self._init_privacy_engine(new_epsilon)

        for epoch in range(EPOCHS):
            self.model.train()
            total_loss = 0
            task_correct = {task: 0 for task in self.criterion_dict.keys()}
            task_total = {task: 0 for task in self.criterion_dict.keys()}

            with BatchMemoryManager(
                    data_loader=self.train_loader,
                    max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
                    optimizer=self.optimizer,
            ) as memory_safe_loader:
                for images, labels in memory_safe_loader:
                    images = images.to(self.device)
                    labels = {k: v.to(self.device) for k, v in labels.items()}

                    self.optimizer.zero_grad()
                    outputs = self.model(images)

                    loss = sum([
                        self.criterion_dict[task](outputs[task], labels[task])
                        for task in self.criterion_dict.keys()
                    ])

                    for task in self.criterion_dict.keys():
                        preds = outputs[task].argmax(dim=1)
                        task_correct[task] += (preds == labels[task]).sum().item()
                        task_total[task] += labels[task].size(0)

                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()

            accuracies = [task_correct[task] / task_total[task] for task in self.criterion_dict.keys()]
            epoch_acc = np.mean(accuracies)
            epoch_loss = total_loss / len(self.train_loader)

            log_message(
                f"Client training - Epoch {epoch + 1}: Loss {epoch_loss:.4f}, Avg Accuracy {epoch_acc:.4f}, Epsilon {self.current_epsilon}")

        return self.get_parameters({}), len(self.train_loader.dataset), {}


class EpsilonAwareStrategy(fl.server.strategy.FedAvg):
    def __init__(self, server_proxy_loader, server_val_loader, test_loader, device, proxy_epochs=3, *args,
                 **kwargs):  # FIXED: Increased proxy epochs
        super().__init__(*args, **kwargs)
        self.server_proxy_loader = server_proxy_loader
        self.server_val_loader = server_val_loader
        self.test_loader = test_loader
        self.device = device
        self.proxy_epochs = proxy_epochs
        self.current_epsilon = None
        self.best_epsilon_history = []
        self.latest_parameters = None

    def configure_fit(self, server_round, parameters, client_manager):
        if server_round == 1:  # FIXED: Explicit initial round handling
            self.current_epsilon = 3
            log_message(f"Initial round - Using default epsilon: {self.current_epsilon}")
        else:
            log_message(f"Round {server_round} - Using selected epsilon: {self.current_epsilon}")

        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        for client, fit_ins in client_instructions:
            fit_ins.config["epsilon"] = str(self.current_epsilon)
        return client_instructions

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
            best_epsilon = self.select_optimal_epsilon(aggregated_ndarrays)
            self.current_epsilon = best_epsilon
            self.best_epsilon_history.append(best_epsilon)
            log_message(f"Round {server_round} - Selected epsilon: {best_epsilon}")

        return aggregated_parameters, aggregated_metrics

    def select_optimal_epsilon(self, parameters_ndarrays):
        base_model = DenseNet121().to(self.device)
        base_model = ModuleValidator.fix(base_model)
        params_dict = zip(base_model.state_dict().keys(), parameters_ndarrays)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        base_model.load_state_dict(state_dict)

        epsilon_results = []
        original_params = torch.cat([p.flatten() for p in base_model.parameters()])  # FIXED: Track parameter changes

        for epsilon in CANDIDATE_EPSILONS:
            model = DenseNet121().to(self.device)
            model = ModuleValidator.fix(model)
            model.load_state_dict(base_model.state_dict())
            optimizer = torch.optim.Adam(model.parameters(), lr=8.672307011698221e-05)

            privacy_engine = PrivacyEngine()
            private_model, private_optimizer, private_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=self.server_proxy_loader,
                epochs=self.proxy_epochs,
                target_epsilon=epsilon,
                target_delta=DELTA,
                max_grad_norm=MAX_GRAD_NORM,
            )
            log_message(f"Epsilon {epsilon} - Using PrivacyEngine with target epsilon: {epsilon}, target delta: {DELTA}, max grad norm: {MAX_GRAD_NORM}")

            private_model.train()
            total_loss = 0.0  # FIXED: Track training progress
            with BatchMemoryManager(
                    data_loader=private_loader,
                    max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
                    optimizer=private_optimizer
            ) as memory_safe_loader:
                for epoch in range(self.proxy_epochs):
                    epoch_loss = 0.0
                    for images, labels in memory_safe_loader:
                        images = images.to(self.device)
                        labels = {k: v.to(self.device) for k, v in labels.items()}

                        private_optimizer.zero_grad()
                        outputs = private_model(images)

                        loss = sum([
                            nn.CrossEntropyLoss()(outputs[task], labels[task])
                            for task in outputs.keys()
                        ])

                        loss.backward()
                        private_optimizer.step()
                        epoch_loss += loss.item()

                    avg_epoch_loss = epoch_loss / len(memory_safe_loader)
                    log_message(f"Epsilon {epsilon} - Proxy Epoch {epoch + 1}: Loss {avg_epoch_loss:.4f}")

            # FIXED: Verify parameter updates
            updated_params = torch.cat([p.flatten() for p in private_model.parameters()])
            param_change = torch.norm(updated_params - original_params).item()
            log_message(f"Epsilon {epsilon} - Parameter change: {param_change:.4f}")

            val_metrics = self._evaluate_model(private_model, self.server_val_loader)
            epsilon_results.append((epsilon, val_metrics))

            log_message(f"Epsilon {epsilon} - Val Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")

            del private_model, private_optimizer, privacy_engine
            torch.cuda.empty_cache()

        test_metrics = self._evaluate_model(base_model, self.test_loader)
        log_message(f"Test Metrics after Round: Acc {test_metrics['accuracy']:.4f}, F1 {test_metrics['f1']:.4f}")

        return self._calculate_best_epsilon(epsilon_results)

    def _evaluate_model(self, model, loader):
        model.eval()
        tasks = ["super_class", "malignancy", "main_class_1", "main_class_2", "sub_class"]
        all_labels = {task: [] for task in tasks}
        all_preds = {task: [] for task in tasks}

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                outputs = model(images)

                for task in tasks:
                    task_labels = labels[task].cpu().numpy()
                    task_preds = outputs[task].argmax(dim=1).cpu().numpy()
                    all_labels[task].extend(task_labels)
                    all_preds[task].extend(task_preds)

        accuracies = []
        f1_scores = []
        for task in tasks:
            task_acc = np.mean(np.array(all_preds[task]) == np.array(all_labels[task]))
            task_f1 = f1_score(all_labels[task], all_preds[task], average='macro')
            accuracies.append(task_acc)
            f1_scores.append(task_f1)

        return {'accuracy': np.mean(accuracies), 'f1': np.mean(f1_scores)}

    def _calculate_best_epsilon(self, epsilon_results):
        if not epsilon_results:
            return CANDIDATE_EPSILONS[5]

        f1_scores = [metrics['f1'] for epsilon, metrics in epsilon_results]
        min_f1 = min(f1_scores)
        max_f1 = max(f1_scores)
        range_f1 = max_f1 - min_f1 if (max_f1 - min_f1) != 0 else 1e-9

        min_eps = min(CANDIDATE_EPSILONS)
        max_eps = max(CANDIDATE_EPSILONS)
        range_eps = max_eps - min_eps if (max_eps - min_eps) != 0 else 1e-9

        scores = []
        for epsilon, metrics in epsilon_results:
            normalized_f1 = (metrics['f1'] - min_f1) / range_f1
            normalized_epsilon = (epsilon - min_eps) / range_eps
            score = (w1 * normalized_f1) - (w2 * normalized_epsilon)
            scores.append((epsilon, score))

        return max(scores, key=lambda x: x[1])[0]


def federated(train_loaders, val_loaders, server_proxy_loader, server_val_loader, test_loader, num_clients, device):
    def client_fn(cid: str) -> fl.client.Client:
        client_index = int(cid)
        client = FlowerClient(train_loaders[client_index], val_loaders[client_index], device)
        return client.to_client()

    strategy = EpsilonAwareStrategy(
        server_proxy_loader=server_proxy_loader,
        server_val_loader=server_val_loader,
        test_loader=test_loader,
        device=device,
        proxy_epochs=3,  # FIXED: Match updated proxy_epochs
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
    )

    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 0.5} if torch.cuda.is_available() else {"num_cpus": 1},
    )

    final_model = DenseNet121().to(device)
    final_model = ModuleValidator.fix(final_model)
    if strategy.latest_parameters is not None:
        ndarrays = fl.common.parameters_to_ndarrays(strategy.latest_parameters)
        state_dict = {k: torch.tensor(v) for k, v in zip(final_model.state_dict().keys(), ndarrays)}
        final_model.load_state_dict(state_dict)
    torch.save(final_model.state_dict(), save_model_path)
    log_message(f"Final model saved with best epsilons: {strategy.best_epsilon_history}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    train_df = pd.read_csv(train_csv_path).sample(frac=0.1, random_state=42)
    test_df = pd.read_csv(test_csv_path).sample(frac=0.1, random_state=42)

    encoder = MultiLabelEncoder()
    columns_to_encode = ["super_class", "malignancy", "main_class_1", "main_class_2", "sub_class"]
    full_train_encoded = encoder.fit_transform(train_df.copy(), columns_to_encode)

    client_df, server_temp_df = train_test_split(full_train_encoded, test_size=0.4, shuffle=True)
    server_proxy_df, server_val_df = train_test_split(server_temp_df, test_size=0.5, shuffle=True)

    test_encoded = encoder.transform(test_df.copy(), columns_to_encode)

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    def create_dataset(df, img_dir, transform):
        return SkinLesionDataset(
            df.copy().reset_index(drop=True),
            img_dir,
            transform
        )


    test_dataset = create_dataset(test_encoded, test_img_dir, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    server_proxy_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    server_val_transform = test_transform

    server_proxy_dataset = create_dataset(server_proxy_df, train_img_dir, server_proxy_transform)
    server_val_dataset = create_dataset(server_val_df, train_img_dir, server_val_transform)

    # FIXED: Add data size verification
    log_message(f"\nData sizes:\n"
                f"- Client data: {len(client_df)}\n"
                f"- Server proxy: {len(server_proxy_df)}\n"
                f"- Server validation: {len(server_val_df)}\n"
                f"- Test data: {len(test_df)}", mode="w")

    # FIXED: Use DPDataLoader for server_proxy_loader
    server_proxy_loader = DPDataLoader(
        dataset=server_proxy_dataset,
        sample_rate=0.1  # Set this to an appropriate value based on your needs
    )
    server_val_loader = DataLoader(server_val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    client_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    client_dataset = create_dataset(client_df, train_img_dir, client_transform)
    split_sizes = [len(client_dataset) // NUM_CLIENTS] * NUM_CLIENTS
    split_sizes[-1] += len(client_dataset) % NUM_CLIENTS
    client_datasets = random_split(client_dataset, split_sizes)

    client_train_datasets = []
    client_val_datasets = []
    for ds in client_datasets:
        train_size = int(0.8 * len(ds))
        val_size = len(ds) - train_size
        client_train, client_val = random_split(ds, [train_size, val_size])
        client_train_datasets.append(client_train)
        client_val_datasets.append(client_val)

    train_loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True) for ds in client_train_datasets]
    val_loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False) for ds in client_val_datasets]

    log_message("Starting federated learning with dynamic epsilon selection...", mode="a")
    federated(train_loaders, val_loaders, server_proxy_loader, server_val_loader, test_loader, NUM_CLIENTS, device)