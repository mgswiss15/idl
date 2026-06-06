import csv
import json
import os
import pandas as pd
from datetime import datetime
from train import main as run_training
from predict import main as run_prediction

def main():
    # 1. Define the experimental matrix you want to benchmark
    target_datasets = {
        "orgs_64": {"CHANNELS": 1, "NUM_CLASSES": 11},
        "cells_64":   {"CHANNELS": 3, "NUM_CLASSES": 8},
        "lesions_64":    {"CHANNELS": 3, "NUM_CLASSES": 7},
        "liver_64":    {"CHANNELS": 1, "NUM_CLASSES": 4}
    }
    
    target_models = ["AlexNet", "VGG", "ResNet"]
    
    # 2. Set up base hyperparameters
    base_config = {
        "BATCH_SIZE": 128,
        "LEARNING_RATE": 0.001,
        "EPOCHS": 10,              # Increased for meaningful evaluation
        "DATA_PATH": "../data/",
        "drop_rate": 0.5,
        "activation": "nn.ReLU(inplace=True)"
    }
    
    results_registry = []
    os.makedirs("./results", exist_ok=True)
    # Header fields for our CSV file
    headers = ["Dataset", "Model", "Train_Accuracy", "Train_Loss", "Val_Accuracy", "Val_Loss", "Test_Accuracy", "Test_F1_Macro", "Runtime_Sec"]
    
    # 3. Execution Grid Loop
    for dataset_name, data_meta in target_datasets.items():
        for model_name in target_models:
            print(f"\n{'='*60}")
            print(f"LAUNCHING: Model={model_name} | Dataset={dataset_name}")
            print(f"{'='*60}")
            
            # Synthesize configuration dictionary state
            current_config = base_config.copy()
            current_config.update({
                "DATA": dataset_name,
                "MODEL": model_name,
                "CHANNELS": data_meta["CHANNELS"],
                "NUM_CLASSES": data_meta["NUM_CLASSES"]
            })
            
            # Overwrite the temporary config.json asset that train/predict rely on
            with open("config.json", "w") as f:
                json.dump(current_config, f, indent=4)
                
            start_time = datetime.now()
            
            try:
                # Execute Training Cycle
                train_loss, train_acc, val_loss, val_acc = run_training(config_path="config.json")
                
                # Execute Evaluation/Prediction Cycle
                report_dict = run_prediction(config_path="config.json")
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Append telemetry to registry
                results_registry.append({
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Train_Accuracy": round(train_acc, 4),
                    "Train_Loss": round(train_loss, 4),
                    "Val_Accuracy": round(val_acc, 4),
                    "Val_Loss": round(val_loss, 4),
                    "Test_Accuracy": round(report_dict['accuracy'], 4),
                    "Test_F1_Macro": round(report_dict['macro avg']['f1-score'], 4),
                    "Runtime_Sec": round(execution_time, 2)
                })
                
            except Exception as e:
                print(f"CRITICAL ERROR processing {model_name} on {dataset_name}: {e}")
                results_registry.append({
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Train_Accuracy": "CRASH", "Train_Loss": "CRASH",
                    "Val_Accuracy": "CRASH", "Val_Loss": "CRASH",
                    "Test_Accuracy": "CRASH", "Test_F1_Macro": "CRASH",
                    "Runtime_Sec": 0.0
                })

    # 4. Export consolidated metrics report using native csv tool
    csv_file_path = "./results/matrix_benchmark_report.csv"
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results_registry)
        
    # 5. Clean Python terminal output formatting (replaces pandas printouts)
    print("\n\n BENCHMARK COMPLETE. SUMMARY MATRIX:")
    print("-" * 90)
    print(f"{headers[0]:<12} | {headers[1]:<10} | {headers[2]:<13} | {headers[3]:<13} | {headers[4]:<13} | {headers[5]:<13} | {headers[6]:<13} | {headers[7]:<13} | {headers[8]}")
    print("-" * 90)
    for row in results_registry:
        print(f"{row['Dataset']:<12} | {row['Model']:<10} | {row['Train_Accuracy']:<13} | {row['Train_Loss']:<13} | {row['Val_Accuracy']:<13} | {row['Val_Loss']:<13} | {row['Test_Accuracy']:<13} | {row['Test_F1_Macro']:<13} | {row['Runtime_Sec']}")
    print("-" * 90)

if __name__ == "__main__":
    main()