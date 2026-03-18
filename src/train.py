# Import modeling libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

# Create a dictionary to store models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Dictionary to store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")    
      
    # Step 1: Train the model
    model.fit(X_train_processed, y_train)
    
    
    # Step 2: Make predictions
    y_pred = model.predict(X_test_processed)
    # Some models require probability predictions for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test_processed)[:,1]
    else:
        y_pred_proba = None
    
        
    # Step 3: Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    else:
        roc_auc = roc_auc_score(y_test, y_pred)
    
    
    # Step 4: Store results
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC-AUC': roc_auc,
        'Model': model,
        'Predictions': y_pred,
        'Probabilities': y_pred_proba
    }
    
    # Print metrics for quick inspection
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")

results_df = pd.DataFrame(results).T[
    ['Accuracy','Precision','Recall','F1 Score','ROC-AUC']
]

results_df

# Identify best model based on ROC-AUC
best_model_name = results_df['ROC-AUC'].idxmax()

print(f"Best Model based on ROC-AUC: {best_model_name}")


for name, res in results.items():
    
    cm = confusion_matrix(y_test, res['Predictions'])
    
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    plt.show()

plt.figure(figsize=(7,5))

for name, res in results.items():
    
    if res['Probabilities'] is not None:
        fpr, tpr, _ = roc_curve(y_test, res['Probabilities'])
        plt.plot(fpr, tpr, label=f"{name}")

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()

plt.show()

