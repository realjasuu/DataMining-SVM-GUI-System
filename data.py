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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
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
        if not self.validate_dataset(self.df):  # Changed from validate_data to validate_dataset
            return
            
        try:
            target = self.select_target_column("Select Target for SVM Classification")
            if not target:
                return
                
            # Enhanced target validation
            if pd.api.types.is_float_dtype(self.df[target]):
                self.show_warning("Invalid Target",
                    "Selected column contains continuous values. Please select a categorical column.")
                return
                
            unique_values = self.df[target].nunique()
            if unique_values > 20:  # Increased threshold slightly but still reasonable
                self.show_warning("Invalid Target",
                    "Selected column has too many unique values. Please select a categorical column.")
                return
                
            # Convert target to categorical if numeric
            if pd.api.types.is_numeric_dtype(self.df[target]):
                self.df[target] = self.df[target].astype('category')
            
            def process_svm():
                try:
                    self.show_status("Preparing data...")
                    X = self.df.drop(columns=[target])
                    y = self.df[target]
                    
                    self.show_status("Scaling features...")
                    X_scaled = self.scaler.fit_transform(X)
                    
                    self.show_status("Splitting data...")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=0.3, random_state=42
                    )
                    
                    self.show_status("Training SVM model...")
                    svm = SVC(kernel=self.kernel_var.get(),
                             C=self.c_var.get(),
                             random_state=42,
                             probability=True)
                    svm.fit(X_train, y_train)
                    
                    self.show_status("Computing predictions...")
                    y_pred = svm.predict(X_test)
                    y_pred_proba = svm.predict_proba(X_test)
                    
                    self.show_status("Calculating metrics...")
                    self.calculate_and_display_metrics(y_test, y_pred, y_pred_proba)
                    
                except Exception as e:
                    self.show_error("SVM Classification Error", str(e))
                finally:
                    self.show_status("Ready")
                    self.stop_progress()
            
            self.start_progress()
            threading.Thread(target=process_svm).start()
            
        except Exception as e:
            self.show_error("Error", f"SVM Classification failed: {str(e)}")

    def calculate_and_display_metrics(self, y_test, y_pred, y_pred_proba):
        """Safely calculate and display metrics"""
        try:
            # Ensure binary classification for ROC
            n_classes = len(np.unique(y_test))
            if n_classes != 2:
                self.show_warning("Metric Calculation",
                    "ROC curve is only available for binary classification. Showing other metrics.")
                
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred)
            
            # Safe specificity calculation
            try:
                specificity = self.calculate_specificity(y_test, y_pred)
            except:
                specificity = None
            
            # Create visualization
            self.display_results(conf_matrix, class_report, specificity,
                               y_test, y_pred_proba if n_classes == 2 else None)
                               
        except Exception as e:
            self.show_error("Metric Calculation Error", str(e))

    def display_results(self, conf_matrix, class_report, specificity, y_test, y_pred_proba=None):
        """Enhanced results display"""
        fig, axes = plt.subplots(1, 3 if y_pred_proba is not None else 2, figsize=(15, 5))
        
        # Confusion Matrix
        sns.heatmap(conf_matrix, annot=True, fmt='d', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        
        # ROC curve if binary classification
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            axes[1].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
            axes[1].plot([0, 1], [0, 1], 'k--')
            axes[1].set_title('ROC Curve')
            axes[1].legend()
            
            metrics_ax = axes[2]
        else:
            metrics_ax = axes[1]
        
        # Metrics text
        metrics_text = f"Classification Report:\n\n{class_report}"
        if specificity is not None:
            metrics_text += f"\nSpecificity: {specificity:.4f}"
            
        metrics_ax.text(0.05, 0.95, metrics_text,
                       fontfamily='monospace', fontsize=10,
                       verticalalignment='top')
        metrics_ax.axis('off')
        
        plt.tight_layout()
        self.show_plot(fig)

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


if __name__ == "__main__":
    root = tk.Tk()
    app = DataMiningSVMGUI(root)
    root.mainloop()