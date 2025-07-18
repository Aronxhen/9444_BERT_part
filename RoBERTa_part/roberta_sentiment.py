"""
RoBERTa for Sentiment Analysis - Improved Implementation

基于论文：RoBERTa: A Robustly Optimized BERT Pretraining Approach  
arXiv: https://arxiv.org/abs/1907.11692

主要改进：
- 使用RoBERTa替代BERT (预期性能提升至98%+)
- 优化的学习率调度策略 (Cosine with warmup)
- 改进的训练参数 (更低学习率，权重衰减)
- 更好的数据预处理 (更长序列支持)
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch.nn as nn
from transformers import (
    RobertaForSequenceClassification, 
    RobertaTokenizer,
    get_cosine_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')


class IMDBDatasetRoBERTa(Dataset):
    """
    优化的IMDB数据集类，支持RoBERTa tokenizer
    """
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # RoBERTa tokenization with improved parameters
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,
            return_attention_mask=True
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class RoBERTaSentimentClassifier:
    """
    RoBERTa情感分析分类器
    """
    def __init__(self, model_name="roberta-base", num_labels=2, max_length=512, batch_size=8):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Initialize model and tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"✅ Model initialized: {model_name}")
        print(f"📱 Device: {self.device}")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def load_data(self, train_path, val_path, test_path):
        """
        加载IMDB数据集
        """
        print("📂 Loading datasets...")
        
        self.df_train = pd.read_csv(train_path)
        self.df_val = pd.read_csv(val_path)
        self.df_test = pd.read_csv(test_path)
        
        print(f"Train samples: {len(self.df_train):,}")
        print(f"Validation samples: {len(self.df_val):,}")
        print(f"Test samples: {len(self.df_test):,}")
        print(f"Total samples: {len(self.df_train) + len(self.df_val) + len(self.df_test):,}")
        
        # Check label distribution
        print("\n📊 Label distribution:")
        print(f"Train: {dict(self.df_train['label'].value_counts().sort_index())}")
        print(f"Val: {dict(self.df_val['label'].value_counts().sort_index())}")
        print(f"Test: {dict(self.df_test['label'].value_counts().sort_index())}")
        
        # Create datasets
        self.train_dataset = IMDBDatasetRoBERTa(self.df_train, self.tokenizer, self.max_length)
        self.val_dataset = IMDBDatasetRoBERTa(self.df_val, self.tokenizer, self.max_length)
        self.test_dataset = IMDBDatasetRoBERTa(self.df_test, self.tokenizer, self.max_length)
        
        # Create dataloaders
        self.train_dataloader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            pin_memory=True,
            num_workers=2
        )
        
        self.val_dataloader = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=2
        )
        
        self.test_dataloader = DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=2
        )
        
        print(f"✅ Dataloaders created with batch_size={self.batch_size}")
    
    def setup_training(self, learning_rate=1e-5, weight_decay=0.01, num_epochs=4, warmup_ratio=0.1):
        """
        设置训练参数
        """
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        
        # Optimizer with weight decay
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=1e-8
        )
        
        # Calculate training steps
        self.num_training_steps = num_epochs * len(self.train_dataloader)
        self.num_warmup_steps = int(warmup_ratio * self.num_training_steps)
        
        # Cosine learning rate scheduler with warmup
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps
        )
        
        print("⚙️ Training setup completed:")
        print(f"Learning rate: {learning_rate}")
        print(f"Weight decay: {weight_decay}")
        print(f"Epochs: {num_epochs}")
        print(f"Training steps: {self.num_training_steps}")
        print(f"Warmup steps: {self.num_warmup_steps}")
    
    def train(self):
        """
        训练模型
        """
        print("\n🚀 Starting RoBERTa training...")
        
        self.model.train()
        best_val_acc = 0
        best_model_state = None
        
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_f1': []
        }
        
        for epoch in range(self.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print(f"{'='*60}")
            
            # Training phase
            self.model.train()
            total_loss = 0
            all_preds = []
            all_labels = []
            
            epoch_start_time = time.time()
            
            train_loop = tqdm(self.train_dataloader, desc="Training", leave=False)
            for step, batch in enumerate(train_loop):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Collect metrics
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(batch['labels'].detach().cpu().numpy())
                
                # Update progress bar
                current_lr = self.scheduler.get_last_lr()[0]
                train_loop.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.2e}'
                })
            
            # Calculate training metrics
            avg_train_loss = total_loss / len(self.train_dataloader)
            train_acc = accuracy_score(all_labels, all_preds)
            train_f1 = f1_score(all_labels, all_preds)
            
            epoch_time = time.time() - epoch_start_time
            
            print(f"\n📈 Training Results:")
            print(f"Loss: {avg_train_loss:.4f} | Accuracy: {train_acc:.4f} | F1: {train_f1:.4f}")
            print(f"Time: {epoch_time:.1f}s")
            
            # Validation phase
            val_acc, val_f1, val_precision, val_recall = self.evaluate(self.val_dataloader, "Validation")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                print(f"✅ New best model saved! Val Accuracy: {val_acc:.4f}")
            
            # Store history
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['val_f1'].append(val_f1)
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\n🎯 Best validation accuracy: {best_val_acc:.4f}")
        
        return self.training_history
    
    @torch.no_grad()
    def evaluate(self, dataloader, phase="Evaluation"):
        """
        评估模型
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        eval_loop = tqdm(dataloader, desc=f"{phase}", leave=False)
        for batch in eval_loop:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        
        print(f"\n📊 {phase} Results:")
        print(f"Accuracy: {accuracy:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
        
        return accuracy, f1, precision, recall
    
    def final_test_evaluation(self):
        """
        最终测试评估
        """
        print("\n🧪 Final Test Evaluation")
        print("="*60)
        
        # Get detailed results
        test_acc, test_f1, test_precision, test_recall = self.evaluate(self.test_dataloader, "Test")
        
        # Get predictions for confusion matrix
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.test_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        # Print final results
        print(f"\n🎯 FINAL TEST RESULTS")
        print(f"✅ Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"📊 F1 Score:  {test_f1:.4f}")
        print(f"🎯 Precision: {test_precision:.4f}")
        print(f"📈 Recall:    {test_recall:.4f}")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        target_names = ['Negative', 'Positive']
        print(classification_report(all_labels, all_preds, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.title('RoBERTa - Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
        
        return test_acc, test_f1, test_precision, test_recall
    
    def plot_training_history(self):
        """
        绘制训练历史
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        
        # Training Loss
        ax1.plot(epochs, self.training_history['train_loss'], 'b-', label='Training Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Training Accuracy
        ax2.plot(epochs, self.training_history['train_acc'], 'b-', label='Training Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Validation Accuracy
        ax3.plot(epochs, self.training_history['val_acc'], 'r-', label='Validation Accuracy')
        ax3.set_title('Validation Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True)
        
        # Validation F1
        ax4.plot(epochs, self.training_history['val_f1'], 'g-', label='Validation F1')
        ax4.set_title('Validation F1 Score')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('F1 Score')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, save_dir="roberta_sentiment_model"):
        """
        保存模型
        """
        # Save model and tokenizer
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(f"{save_dir}_tokenizer")
        
        # Save state dict
        torch.save(self.model.state_dict(), f"{save_dir}.pt")
        
        print(f"✅ Model saved to: {save_dir}")
        print(f"✅ Tokenizer saved to: {save_dir}_tokenizer")
        print(f"✅ State dict saved to: {save_dir}.pt")
    
    def predict_text(self, text):
        """
        预测单个文本的情感
        """
        self.model.eval()
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)
        
        sentiment = "Positive" if pred.item() == 1 else "Negative"
        confidence = probs[0][pred.item()].item()
        
        return sentiment, confidence


def main():
    """
    主函数 - 完整的训练流程
    """
    print("🤖 RoBERTa Sentiment Analysis")
    print("="*50)
    
    # Initialize classifier
    classifier = RoBERTaSentimentClassifier(
        model_name="roberta-base",
        max_length=512,
        batch_size=8
    )
    
    # Load data
    classifier.load_data(
        train_path="content/imdb_train.csv",
        val_path="content/imdb_validation.csv", 
        test_path="content/imdb_test.csv"
    )
    
    # Setup training
    classifier.setup_training(
        learning_rate=1e-5,
        weight_decay=0.01,
        num_epochs=4,
        warmup_ratio=0.1
    )
    
    # Train model
    history = classifier.train()
    
    # Final evaluation
    test_results = classifier.final_test_evaluation()
    
    # Plot training history
    classifier.plot_training_history()
    
    # Save model
    classifier.save_model("roberta_sentiment_final")
    
    # Performance summary
    print(f"\n🏆 PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Model: RoBERTa-base")
    print(f"Dataset: IMDB Sentiment Analysis")
    print(f"Final Test Accuracy: {test_results[0]:.4f} ({test_results[0]*100:.2f}%)")
    print(f"Final Test F1 Score: {test_results[1]:.4f}")
    print("="*50)
    
    # Test prediction
    print(f"\n🧪 Sample Prediction:")
    sample_text = "This movie is absolutely amazing! Great acting and storyline."
    sentiment, confidence = classifier.predict_text(sample_text)
    print(f"Text: {sample_text}")
    print(f"Prediction: {sentiment} (Confidence: {confidence:.4f})")


if __name__ == "__main__":
    main()