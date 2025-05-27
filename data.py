import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                           precision_score, recall_score, f1_score, accuracy_score,
                           precision_recall_curve)  # Add this import
import warnings
from pandas.errors import EmptyDataError
import gc
import threading
import os

class DataMiningSVMGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SVM Classification GUI")
        self.root.geometry("1000x600")

        self.df = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # Initialize feature selection variables first
        self.feature1_var = tk.StringVar()
        self.feature2_var = tk.StringVar()

        # Create Paned Window
        self.paned = tk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=1)

        # Left frame for buttons
        self.left_frame = tk.Frame(self.paned, width=250, padx=10, pady=10)
        self.paned.add(self.left_frame)

        # Right frame for dataset display
        self.right_frame = tk.Frame(self.paned)
        self.paned.add(self.right_frame)

        # Add status bar
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = ttk.Label(self.status_bar, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.progress = ttk.Progressbar(self.status_bar, mode='indeterminate')
        self.progress.pack(side=tk.RIGHT, padx=5)
        
        # Create main sections
        self.create_left_panel()
        self.create_right_panel()

        self.chunk_size = 1000  # For chunked processing
        self.loading_thread = None
        self.processing_queue = []

    def create_left_panel(self):
        """Creates the left panel with controls"""
        # Load CSV section
        load_frame = ttk.LabelFrame(self.left_frame, text="Data Loading", padding=5)
        load_frame.pack(fill='x', pady=5)
        
        load_btn = ttk.Button(load_frame, text="Load CSV", command=self.load_csv)
        load_btn.pack(fill='x', pady=5)
        self.create_tooltip(load_btn, "Load a CSV dataset for analysis")
        
        # Feature Selection section
        feature_frame = ttk.LabelFrame(self.left_frame, text="Hyperplane Section", padding=5)
        feature_frame.pack(fill='x', pady=5)
        
        ttk.Label(feature_frame, text="Feature 1:").pack(fill='x', pady=2)
        self.feature1_combo = ttk.Combobox(feature_frame, textvariable=self.feature1_var, state='readonly')
        self.feature1_combo.pack(fill='x', pady=2)
        
        ttk.Label(feature_frame, text="Feature 2:").pack(fill='x', pady=2)
        self.feature2_combo = ttk.Combobox(feature_frame, textvariable=self.feature2_var, state='readonly')
        self.feature2_combo.pack(fill='x', pady=2)
        
        # Add Visualize Features button
        viz_btn = ttk.Button(feature_frame, text="Visualize Features", 
                          command=lambda: self.run_in_thread(self.visualize_features))
        viz_btn.pack(fill='x', pady=5)
        
        # SVM Parameters section
        svm_frame = ttk.LabelFrame(self.left_frame, text="SVM Parameters", padding=5)
        svm_frame.pack(fill='x', pady=5)
        
        # Kernel selection
        self.kernel_var = tk.StringVar(value='rbf')
        kernel_label = ttk.Label(svm_frame, text="Kernel:")
        kernel_label.pack(fill='x', pady=2)
        kernel_combo = ttk.Combobox(svm_frame, textvariable=self.kernel_var, 
                                  values=['linear', 'rbf', 'poly', 'sigmoid'],
                                  state='readonly')
        kernel_combo.pack(fill='x', pady=2)
        
        # C parameter
        self.c_var = tk.DoubleVar(value=1.0)
        c_label = ttk.Label(svm_frame, text="C (Regularization):")
        c_label.pack(fill='x', pady=2)
        c_entry = ttk.Entry(svm_frame, textvariable=self.c_var)
        c_entry.pack(fill='x', pady=2)
        
        # Run button
        run_btn = ttk.Button(svm_frame, text="Run SVM Classification", 
                            command=lambda: self.run_in_thread(self.run_svm_classification))
        run_btn.pack(fill='x', pady=5)

    def create_right_panel(self):
        """Creates the right panel with dataset display"""
        # Configure grid
        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)
        
        # Create treeview with scrollbars
        self.tree = ttk.Treeview(self.right_frame, show='headings')
        self.tree.grid(row=0, column=0, sticky='nsew')
        
        # Scrollbars
        y_scroll = ttk.Scrollbar(self.right_frame, orient="vertical", command=self.tree.yview)
        y_scroll.grid(row=0, column=1, sticky='ns')
        
        x_scroll = ttk.Scrollbar(self.right_frame, orient="horizontal", command=self.tree.xview)
        x_scroll.grid(row=1, column=0, sticky='ew')
        
        self.tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

    def show_status(self, message):
        self.status_label.config(text=message)
        self.root.update()

    def start_progress(self):
        self.progress.start(10)
        self.root.update()

    def stop_progress(self):
        self.progress.stop()
        self.root.update()

    def validate_dataset(self, df):
        if df.empty:
            raise EmptyDataError("Dataset is empty")
        if df.shape[0] < 2:
            raise ValueError("Dataset must have at least 2 rows")
        if df.shape[1] < 2:
            raise ValueError("Dataset must have at least 2 columns")
        return True

    def load_csv(self):
        path = filedialog.askopenfilename(
            filetypes=[('CSV Files', '*.csv'), ('All Files', '*.*')]
        )
        if not path:
            return
            
        def process_in_background():
            try:
                self.start_progress()
                self.show_status("Analyzing file size...")
                
                # First check file size
                file_size = os.path.getsize(path) / (1024 * 1024)  # Size in MB
                if file_size > 100:  # If file is larger than 100MB
                    if not messagebox.askyesno("Large File Warning",
                        f"The selected file is {file_size:.1f}MB. Loading large files may take time. Continue?"):
                        return
                
                # Use chunks for large files
                chunks = pd.read_csv(path, chunksize=self.chunk_size, low_memory=False)
                self.df = next(chunks)  # Load first chunk
                
                self.show_status("Processing data types...")
                # Analyze column types from first chunk
                categorical_cols = []
                numeric_cols = []
                
                for col in self.df.columns:
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        numeric_cols.append(col)
                    else:
                        categorical_cols.append(col)
                
                # Process rest of chunks in background
                def process_chunks():
                    try:
                        total_rows = 0
                        for chunk in chunks:
                            self.df = pd.concat([self.df, chunk])
                            total_rows += len(chunk)
                            self.show_status(f"Processed {total_rows} rows...")
                            
                            # Handle memory
                            if total_rows % (self.chunk_size * 10) == 0:
                                gc.collect()
                                
                        self.post_load_processing(categorical_cols, numeric_cols)
                    except Exception as e:
                        self.show_error("Error processing file", str(e))
                
                threading.Thread(target=process_chunks).start()
                self.display_dataset()  # Show first chunk immediately
                
            except Exception as e:
                self.show_error("Error loading file", str(e))
            finally:
                self.stop_progress()
        
        self.loading_thread = threading.Thread(target=process_in_background)
        self.loading_thread.start()

    def post_load_processing(self, categorical_cols, numeric_cols):
        """Handle post-load data processing"""
        try:
            self.show_status("Processing missing values...")
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
            self.df[categorical_cols] = self.df[categorical_cols].fillna('Unknown')
            
            self.show_status("Encoding categorical variables...")
            for col in categorical_cols:
                try:
                    self.df[col] = self.label_encoder.fit_transform(self.df[col].astype(str))
                except Exception as e:
                    self.show_warning(f"Could not encode column {col}", str(e))
            
            # Update feature selection dropdowns
            all_features = list(self.df.columns)
            self.feature1_combo['values'] = all_features
            self.feature2_combo['values'] = all_features
            
            # Set default features if available
            if 'age' in all_features and 'hrs_sitting' in all_features:
                self.feature1_var.set('age')
                self.feature2_var.set('hrs_sitting')
            else:
                self.feature1_var.set(all_features[0])
                self.feature2_var.set(all_features[1] if len(all_features) > 1 else all_features[0])
            
            self.show_status("Ready")
            self.root.update()
        except Exception as e:
            self.show_error("Error in post-processing", str(e))

    def show_error(self, title, message):
        """Unified error display"""
        messagebox.showerror(title, message)
        self.show_status("Error occurred")
        
    def show_warning(self, title, message):
        """Unified warning display"""
        messagebox.showwarning(title, message)
        
    def run_svm_classification(self):
        if not self.validate_dataset(self.df):
            return
            
        try:
            target = self.select_target_column("Select Target for SVM Classification")
            if not target:
                return

            # Store target name for display
            self.current_target = target
                
            # Enhanced target validation
            if pd.api.types.is_float_dtype(self.df[target]):
                self.show_warning("Invalid Target",
                    "Selected column contains continuous values. Please select a categorical column.")
                return
                
            unique_values = self.df[target].nunique()
            if unique_values > 20:
                self.show_warning("Invalid Target",
                    "Selected column has too many unique values. Please select a categorical column.")
                return
                
            # Convert target to categorical if numeric
            if pd.api.types.is_numeric_dtype(self.df[target]):
                self.df[target] = self.df[target].astype('category')
            
            def process_svm():
                try:
                    self.show_status("Preparing data...")
                    X = self.df[[self.feature1_var.get(), self.feature2_var.get()]]
                    y = self.df[target]
                    
                    # Scale features
                    X_scaled = self.scaler.fit_transform(X)
                    
                    self.show_status("Training SVM model...")
                    svm = SVC(kernel=self.kernel_var.get(),
                            C=self.c_var.get(),
                            random_state=42,
                            probability=True)
                    svm.fit(X_scaled, y)
                    
                    self.show_status("Computing metrics...")
                    y_pred = svm.predict(X_scaled)
                    y_proba = svm.predict_proba(X_scaled)
                    
                    # Display comprehensive results in single window
                    self.display_full_results(X_scaled, y, y_pred, y_proba, svm, target)
                    
                except Exception as e:
                    self.show_error("SVM Error", str(e))
                finally:
                    self.stop_progress()
            
            self.run_in_thread(process_svm)
            
        except Exception as e:
            self.show_error("Error", str(e))

    def display_full_results(self, X, y_true, y_pred, y_proba, model, target_name):
        """Display CM, ROC, PR (Lift), and classification report in one figure"""
        try:
            conf_matrix = confusion_matrix(y_true, y_pred)
            class_report = classification_report(y_true, y_pred)
            n_classes = len(np.unique(y_true))

            try:
                specificity = self.calculate_specificity(y_true, y_pred)
            except:
                specificity = None

            # Create figure based on number of classes
            if n_classes == 2:
                # For binary classification: 2x2 subplot
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                axes = axes.flatten()  # Convert to 1D array for easier indexing
            else:
                # For multiclass: Use single row with 2 plots
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                axes = [axes[0], axes[1], None, None]  # Pad with None for consistent indexing

            # Confusion Matrix (always show)
            sns.heatmap(conf_matrix, annot=True, fmt='d', ax=axes[0], cmap='Blues')
            axes[0].set_title('Confusion Matrix')

            # Classification Report (always show)
            metrics_text = f"SVM Classification Report\nTarget: {target_name}\n\n{class_report}"
            if specificity is not None:
                metrics_text += f"\nSpecificity: {specificity:.4f}"
            axes[1].text(0.05, 0.95, metrics_text,
                        fontfamily='monospace', fontsize=10,
                        verticalalignment='top')
            axes[1].axis('off')

            # ROC and Lift curves only for binary classification
            if n_classes == 2 and axes[2] is not None:
                # ROC Curve
                fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                axes[2].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
                axes[2].plot([0, 1], [0, 1], 'k--')
                axes[2].set_title('ROC Curve')
                axes[2].set_xlabel('False Positive Rate')
                axes[2].set_ylabel('True Positive Rate')
                axes[2].legend()

                # Lift Curve
                precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
                axes[3].plot(recall, precision, label='Lift Curve', color='purple')
                axes[3].set_title('Lift Curve (Precision vs Recall)')
                axes[3].set_xlabel('Recall')
                axes[3].set_ylabel('Precision')
                axes[3].legend()

            plt.tight_layout()
            self.show_plot(fig, additional_info="SVM Evaluation Summary")

        except Exception as e:
            self.show_error("Visualization Error", str(e))

    def plot_decision_boundary(self, X, y, clf, feature_names, title=None):
        """Plot SVM decision boundary and margins"""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create mesh grid for decision boundary
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 200),
                np.linspace(y_min, y_max, 200)
            )
            
            # Plot decision regions
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
            
            # Plot decision boundary and margins
            if hasattr(clf, 'decision_function'):
                Z_decision = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
                Z_decision = Z_decision.reshape(xx.shape)
                ax.contour(xx, yy, Z_decision, colors='k',
                          levels=[-1, 0, 1], alpha=0.5,
                          linestyles=['--', '-', '--'])
            
            # Plot support vectors
            if hasattr(clf, 'support_vectors_'):
                ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                          s=100, linewidth=1, facecolors='none', edgecolors='k',
                          label='Support Vectors')
            
            # Plot training data
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu',
                               alpha=0.8, label='Training Data')
            plt.colorbar(scatter)
            
            ax.set_xlabel(feature_names[0])
            ax.set_ylabel(feature_names[1])
            ax.set_title(title or 'SVM Decision Boundary')
            ax.legend()
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.show_error("Visualization Error", str(e))
            return None

    def visualize_features(self):
        """Feature visualization with maximized margin hyperplane and metrics"""
        if not self.validate_dataset(self.df):
            return
            
        try:
            feature1 = self.feature1_var.get()
            feature2 = self.feature2_var.get()
            
            if not feature1 or not feature2:
                self.show_warning("Feature Selection", "Please select two features")
                return
            
            # Ask user to select target column manually
            target = self.select_target_column("Select Target for Visualization")
            
            if not target:
                return

            # Ensure binary classification
            if self.df[target].nunique() != 2:
                self.show_warning("Visualization Error", "Target must have exactly 2 unique classes.")
                return
            
            # Prepare data
            X = self.df[[feature1, feature2]].values
            y = self.df[target]
            
            # Scale features for better SVM performance
            X_scaled = self.scaler.fit_transform(X)
            
            # Train linear SVM with C=1.0 for clear margin visualization
            svm = SVC(kernel='linear', C=1.0, random_state=42, probability=True)  # Add probability=True
            svm.fit(X_scaled, y)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot data points first
            classes = np.unique(y)
            colors = ['#2ecc71', '#e74c3c']  # Green for class 0, Red for class 1
            for i, cls in enumerate(classes):
                mask = y == cls
                ax.scatter(X_scaled[mask, 0], X_scaled[mask, 1],
                          c=colors[i], label=f'Class {cls}',
                          alpha=0.6, s=100)
            
            # Get hyperplane coefficients
            w = svm.coef_[0]
            b = svm.intercept_[0]
            margin = 1 / np.sqrt(np.sum(w ** 2))
            
            # Create mesh grid for decision boundary
            x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
            y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                               np.linspace(y_min, y_max, 100))
            
            # Plot decision boundary (hyperplane)
            Z = (w[0] * xx + w[1] * yy + b)
            
            # Plot hyperplane (solid line)
            ax.contour(xx, yy, Z, levels=[0], colors='k', linestyles='-', linewidths=2,
                      label='Hyperplane')
            
            # Plot margins (dashed lines)
            ax.contour(xx, yy, Z, levels=[-1, 1], colors='k', linestyles='--', linewidths=1)
            
            # Highlight support vectors
            ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                      s=300, linewidth=2, facecolors='none', edgecolors='k',
                      label='Support Vectors')
            
            # Add margin width annotation
            ax.text(0.02, 0.98, f'Margin Width: {margin:.3f}',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round',
                                                   facecolor='white',
                                                   alpha=0.8))
            
            # Customize plot
            ax.set_xlabel(f'Scaled {feature1}', fontsize=12)
            ax.set_ylabel(f'Scaled {feature2}', fontsize=12)
            ax.set_title('SVM Maximum Margin Hyperplane\n' + 
                        f'Features: {feature1} vs {feature2}\n' +
                        f'Target: {target}',
                        fontsize=14, pad=20)
            
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            self.show_plot(fig, "SVM Maximum Margin Analysis")
            
            # Add metrics calculation and display
            y_pred = svm.predict(X_scaled)
            y_proba = svm.predict_proba(X_scaled)
            self.show_metrics(target, y, y_pred, y_proba, svm)
            
        except Exception as e:
            self.show_error("Visualization Error", str(e))

    def create_svm_visualization(self, X, y, svm_model, feature_names, target_name):
        """Create enhanced SVM visualization with clear decision boundary and margins"""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create mesh grid for decision boundary
            margin = 0.5
            x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
            y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
            
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 200),
                np.linspace(y_min, y_max, 200)
            )
            
            # Get predictions and decision function values
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            Z = svm_model.predict(mesh_points)
            Z = Z.reshape(xx.shape)
            
            # Plot decision regions with transparent colors
            ax.contourf(xx, yy, Z, alpha=0.2, cmap='RdYlGn')
            
            # Plot decision boundary and margins
            if hasattr(svm_model, 'decision_function'):
                Z_decision = svm_model.decision_function(mesh_points)
                Z_decision = Z_decision.reshape(xx.shape)
                
                # Plot decision boundary (solid line)
                ax.contour(xx, yy, Z_decision, levels=[0],
                          colors='k', linestyles='-', linewidths=2)
                
                # Plot margins (dashed lines)
                ax.contour(xx, yy, Z_decision, levels=[-1, 1],
                          colors='k', linestyles='--', linewidths=1)
                
                # Add margin width annotation
                if svm_model.kernel == 'linear':
                    w = svm_model.coef_[0]
                    margin_width = 2 / np.sqrt(np.sum(w ** 2))
                    ax.text(0.02, 0.98, f'Margin Width: {margin_width:.3f}',
                           transform=ax.transAxes, fontsize=10,
                           verticalalignment='top', bbox=dict(boxstyle='round',
                                                           facecolor='white',
                                                           alpha=0.8))
            
            # Plot support vectors with distinct marking
            if hasattr(svm_model, 'support_vectors_'):
                ax.scatter(svm_model.support_vectors_[:, 0],
                          svm_model.support_vectors_[:, 1],
                          s=300, linewidth=1, facecolors='none',
                          edgecolors='k', label='Support Vectors')
            
            # Plot training data points
            classes = np.unique(y)
            colors = ['g', 'r'] if len(classes) == 2 else plt.cm.rainbow(np.linspace(0, 1, len(classes)))
            for i, cls in enumerate(classes):
                mask = y == cls
                ax.scatter(X[mask, 0], X[mask, 1], c=[colors[i]],
                          label=f'Class {cls}', alpha=0.6, s=100)
            
            # Customize plot
            ax.set_xlabel(feature_names[0], fontsize=12)
            ax.set_ylabel(feature_names[1], fontsize=12)
            ax.set_title(f'SVM Decision Boundary\nKernel: {svm_model.kernel}, C: {svm_model.C}',
                        fontsize=14, pad=20)
            
            # Add legend
            ax.legend(loc='upper right', fontsize=10)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.show_error("Visualization Error", str(e))
            return None

    def calculate_specificity(self, y_true, y_pred):
        """Calculate specificity from confusion matrix"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)

    def show_metrics(self, target_name, y_true, y_pred, y_proba, model):
        """Display SVM metrics in a separate window"""
        try:
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            specificity = self.calculate_specificity(y_true, y_pred)
            
            # Calculate AUC for binary classification
            auc_score = None
            if len(np.unique(y_true)) == 2:
                fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                auc_score = auc(fpr, tpr)
            
            # Create metrics window
            metrics_window = tk.Toplevel(self.root)
            metrics_window.title("SVM Metrics")
            metrics_window.geometry("400x300")
            
            # Create metrics text
            metrics_text = f"SVM Classification Metrics\n"
            metrics_text += f"Target Variable: {target_name}\n\n"
            metrics_text += f"Accuracy:    {accuracy:.4f}\n"
            metrics_text += f"Precision:   {precision:.4f}\n"
            metrics_text += f"Recall:      {recall:.4f}\n"
            metrics_text += f"F1-Score:    {f1:.4f}\n"
            metrics_text += f"Specificity: {specificity:.4f}\n"
            if auc_score is not None:
                metrics_text += f"AUC:         {auc_score:.4f}\n"
            
            # Display metrics
            metrics_label = ttk.Label(metrics_window, text=metrics_text,
                                    font=('Courier', 12), justify='left')
            metrics_label.pack(padx=20, pady=20)
            
            # Add close button
            ttk.Button(metrics_window, text="Close",
                      command=metrics_window.destroy).pack(pady=10)
            
        except Exception as e:
            self.show_error("Metric Calculation Error", str(e))

    def show_plot(self, fig, additional_info=None):
        """Helper to show plots in Tkinter window"""
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Analysis Result")
        
        if additional_info:
            info_label = ttk.Label(plot_window, text=additional_info, justify='left')
            info_label.pack(padx=10, pady=5)
        
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        def on_close():
            plt.close(fig)
            plot_window.destroy()
            
        plot_window.protocol("WM_DELETE_WINDOW", on_close)

    def create_tooltip(self, widget, text):
        """Create a tooltip for a given widget with the given text."""
        def show_tooltip(event=None):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")

            label = ttk.Label(tooltip, text=text, justify='left',
                           background="#ffffe0", relief='solid', borderwidth=1)
            label.pack()

            def hide_tooltip(event=None):
                tooltip.destroy()

            # Schedule removal of tooltip after 2 seconds
            tooltip.after(2000, hide_tooltip)
            
            # Remove tooltip if mouse leaves widget
            widget.bind('<Leave>', hide_tooltip)
            tooltip.bind('<Leave>', hide_tooltip)

        widget.bind('<Enter>', show_tooltip)

    def display_dataset(self):
        """Display the dataset in the treeview"""
        try:
            self.tree.delete(*self.tree.get_children())  # Clear existing items
            
            if self.df is None or self.df.empty:
                return
                
            # Configure columns
            self.tree["columns"] = list(self.df.columns)
            for col in self.df.columns:
                self.tree.heading(col, text=col)
                # Calculate column width based on header and content
                max_width = max(
                    len(str(col)),
                    self.df[col].astype(str).str.len().max()
                ) * 10
                self.tree.column(col, width=min(max_width, 300), anchor="w")
            
            # Display first 1000 rows for performance
            display_df = self.df.head(1000)
            for idx, row in display_df.iterrows():
                self.tree.insert("", "end", values=list(row))
                
            if len(self.df) > 1000:
                self.show_status(f"Displaying first 1000 of {len(self.df)} rows")
                
        except Exception as e:
            self.show_error("Display Error", str(e))

    def run_in_thread(self, func):
        """Execute a function in a separate thread with progress indication"""
        def wrapper():
            try:
                self.start_progress()
                func()
            finally:
                self.stop_progress()
                
        thread = threading.Thread(target=wrapper)
        thread.daemon = True  # Make thread daemon so it doesn't block program exit
        thread.start()

    def select_target_column(self, title):
        """Opens a dialog for selecting a target column"""
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("400x400")
        dialog.minsize(400, 300)
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="20 10 20 10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        header = ttk.Label(main_frame, 
                          text="Select target column for classification:",
                          font=('TkDefaultFont', 10, 'bold'))
        header.pack(pady=(0, 15))
        
        # Modified categorical column selection - more permissive
        categorical_cols = []
        for col in self.df.columns:
            unique_vals = self.df[col].nunique()
            is_numeric = pd.api.types.is_numeric_dtype(self.df[col])
            
            # Include all columns except those with too many unique values
            if unique_vals <= 50 or not is_numeric:  # Increased threshold
                categorical_cols.append(col)
        
        if not categorical_cols:
            messagebox.showwarning("Warning", 
                "No suitable categorical columns found.\nTarget column must be categorical or have few unique values.")
            dialog.destroy()
            return None
        
        # Combobox frame
        combo_frame = ttk.Frame(main_frame)
        combo_frame.pack(fill=tk.X, pady=(0, 15))
        
        var = tk.StringVar(value=categorical_cols[0])
        combo = ttk.Combobox(combo_frame, textvariable=var, 
                            values=categorical_cols, state='readonly', 
                            width=40)
        combo.pack(side=tk.LEFT, expand=True)
        
        # Info section with scrollable text
        info_frame = ttk.LabelFrame(main_frame, text="Column Information", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Create scrollable text widget for column info
        info_text = tk.Text(info_frame, height=10, width=40, wrap=tk.WORD)
        info_scroll = ttk.Scrollbar(info_frame, orient="vertical", 
                                  command=info_text.yview)
        info_text.configure(yscrollcommand=info_scroll.set)
        
        info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Enhanced column information display
        column_info = []
        for col in categorical_cols:
            dtype = self.df[col].dtype
            n_unique = self.df[col].nunique()
            n_missing = self.df[col].isna().sum()
            info = f"{col}:\n  Type: {dtype}\n  Unique values: {n_unique}\n  Missing: {n_missing}\n"
            column_info.append(info)
        
        info_text.insert("1.0", "\n".join(column_info))
        info_text.configure(state="disabled")  # Make read-only
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        result = [None]
        
        def on_ok():
            result[0] = var.get()
            dialog.destroy()
            
        ttk.Button(button_frame, text="OK", command=on_ok, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy, width=15).pack(side=tk.RIGHT, padx=5)
        
        # Center the dialog on screen
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'+{x}+{y}')
        
        dialog.wait_window()
        return result[0]

    def visualize_decision_boundary(self, X, y, svm_model, feature_names):
        """Visualize SVM decision boundary with two features"""
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Create mesh grid
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                               np.linspace(y_min, y_max, 200))
            
            # Get predictions for mesh grid points
            Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot decision regions
            ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
            
            # Plot decision boundary and margins
            if hasattr(svm_model, 'decision_function'):
                Z_decision = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
                Z_decision = Z_decision.reshape(xx.shape)
                ax.contour(xx, yy, Z_decision, colors='k',
                          levels=[-1, 0, 1], alpha=0.5,
                          linestyles=['--', '-', '--'])
            
            # Plot support vectors if available
            if hasattr(svm_model, 'support_vectors_'):
                ax.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1],
                          s=100, facecolors='none', edgecolors='k',
                          label='Support Vectors')
            
            # Plot data points
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu',
                               label='Data Points')
            plt.colorbar(scatter)
            
            # Set labels and title
            ax.set_xlabel(feature_names[0])
            ax.set_ylabel(feature_names[1])
            ax.set_title('SVM Decision Boundary')
            ax.legend()
            
            return fig
            
        except Exception as e:
            self.show_error("Visualization Error", str(e))
            return None

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = DataMiningSVMGUI(root)
        root.protocol("WM_DELETE_WINDOW", root.quit)  # Handle window closing
        root.mainloop()
    except Exception as e:
        print(f"Error starting application: {str(e)}")
        import traceback
        traceback.print_exc()