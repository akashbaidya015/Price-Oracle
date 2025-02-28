import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

class IPhonePricePredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("iPhone Price Predictor")
        self.root.geometry("1200x800")

        self.df = None
        self.model = None
        self.current_column = None

        self.setup_gui()

    def setup_gui(self):
        """Creates the UI elements."""
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)

        ttk.Button(control_frame, text="Load Data", command=self.load_data).pack(side=tk.LEFT, padx=5)

        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(control_frame, textvariable=self.model_var, state='disabled')
        self.model_dropdown.pack(side=tk.LEFT, padx=5)
        self.model_dropdown.bind('<<ComboboxSelected>>', self.on_model_select)

        ttk.Button(control_frame, text="Predict 6 Months", command=self.predict).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Predict Specific Date", command=self.predict_specific_date).pack(side=tk.LEFT, padx=5)  # New Feature!

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(control_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=5)

        self.fig = Figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def load_data(self):
        """Handles file selection and data loading."""
        try:
            filename = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
            if filename:
                self.df = pd.read_csv(filename)
                self.df['date'] = pd.to_datetime(self.df['date'])

                price_columns = [col for col in self.df.columns if col != 'date']
                self.model_dropdown['values'] = price_columns
                self.model_dropdown['state'] = 'readonly'

                self.status_var.set("Data loaded successfully")
                messagebox.showinfo("Success", "Data loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading data: {str(e)}")

    def on_model_select(self, event=None):
        """Updates the selected iPhone model and plots its data."""
        if self.model_var.get():
            self.current_column = self.model_var.get()
            self.plot_data()

    def prepare_data(self, data, column):
        """Prepares the dataset for training."""
        data = data.dropna(subset=[column]).copy()
        first_date = data['date'].min()
        data['days'] = (data['date'] - first_date).dt.days
        return data

    def train_model(self, X, y):
        """Trains a polynomial regression model."""
        model = make_pipeline(
            PolynomialFeatures(degree=3, include_bias=False),
            Ridge(alpha=0.1)  # Ridge regression to prevent overfitting
        )
        model.fit(X.reshape(-1, 1), y)
        return model

    def plot_data(self):
        """Plots historical price data."""
        if self.df is None or self.current_column is None:
            return
            
        self.ax.clear()

        valid_data = self.df[self.df[self.current_column].notna()]
        self.ax.scatter(valid_data['date'], valid_data[self.current_column], label='Historical Data', alpha=0.5)

        self.ax.set_title(f'{self.current_column} Price History')
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Price ($)')
        self.ax.grid(True)
        self.fig.autofmt_xdate()

        self.canvas.draw()

    def predict(self):
        """Predicts the next 6 months of prices and updates the graph."""
        if self.current_column is None:
            messagebox.showwarning("Warning", "Please select an iPhone model first!")
            return
            
        try:
            clean_data = self.prepare_data(self.df, self.current_column)
            if len(clean_data) < 10:
                messagebox.showwarning("Warning", "Not enough price data for prediction!")
                return
                
            X = clean_data['days'].values
            y = clean_data[self.current_column].values

            model = self.train_model(X, y)

            last_date = clean_data['date'].max()
            future_dates = pd.date_range(start=last_date, periods=180, freq='D')
            first_date = clean_data['date'].min()
            future_days = (future_dates - first_date).days

            future_prices = model.predict(future_days.values.reshape(-1, 1))

            self.ax.clear()
            self.ax.scatter(clean_data['date'], clean_data[self.current_column], label='Historical Data', alpha=0.5)
            self.ax.plot(future_dates, future_prices, 'r--', label='6-Month Prediction')

            self.ax.set_title(f'{self.current_column} - Historical Data and 6-Month Prediction')
            self.ax.set_xlabel('Date')
            self.ax.set_ylabel('Price ($)')
            self.ax.legend()
            self.ax.grid(True)
            self.fig.autofmt_xdate()

            self.canvas.draw()

            avg_price = np.mean(future_prices)
            max_price = np.max(future_prices)
            min_price = np.min(future_prices)

            messagebox.showinfo(
                "Prediction Summary",
                f"6-Month Price Prediction Summary:\n\n"
                f"Average Price: ${avg_price:.2f}\n"
                f"Maximum Price: ${max_price:.2f}\n"
                f"Minimum Price: ${min_price:.2f}"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Error making prediction: {str(e)}")

    def predict_specific_date(self):
        """Allows the user to enter a date and predicts the price for that date."""
        if self.current_column is None:
            messagebox.showwarning("Warning", "Please select an iPhone model first!")
            return

        try:
            user_date = simpledialog.askstring("Enter Date", "Enter a future date (YYYY-MM-DD):")

            if not user_date:
                messagebox.showinfo("Cancelled", "Prediction cancelled.")
                return  

            try:
                user_date = datetime.strptime(user_date, "%Y-%m-%d")
            except ValueError:
                messagebox.showerror("Invalid Date", "Please enter a valid date in YYYY-MM-DD format.")
                return

            clean_data = self.prepare_data(self.df, self.current_column)
            if len(clean_data) < 10:
                messagebox.showwarning("Warning", "Not enough price data for prediction!")
                return
                
            X = clean_data['days'].values
            y = clean_data[self.current_column].values

            model = self.train_model(X, y)

            first_date = clean_data['date'].min()
            user_days = (user_date - first_date).days

            predicted_price = model.predict(np.array([[user_days]]))[0]

            messagebox.showinfo("Predicted Price", f"Predicted price for {user_date.strftime('%Y-%m-%d')}: ${predicted_price:.2f}")

        except Exception as e:
            messagebox.showerror("Error", f"Error making prediction: {str(e)}")

def main():
    root = tk.Tk()
    app = IPhonePricePredictorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting gracefully...")
