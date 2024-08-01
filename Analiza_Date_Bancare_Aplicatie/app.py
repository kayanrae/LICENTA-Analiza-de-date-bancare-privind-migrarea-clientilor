import sys
import pandas as pd
import numpy as np
import plotly.express as px
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QTableView, QWidget, QHeaderView, QFileDialog, QLabel, QMessageBox, QStackedWidget
from PyQt5.QtCore import QAbstractTableModel, Qt, QModelIndex
from PyQt5.QtWebEngineWidgets import QWebEngineView
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

class PandasModel(QAbstractTableModel):
    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if index.isValid():
            row = index.row()
            col = index.column()
            self._data.iat[row, col] = value
            self.dataChanged.emit(index, index, (Qt.DisplayRole,))
            return True
        return False

    def flags(self, index):
        return Qt.ItemIsSelectable | Qt.ItemIsEditable | Qt.ItemIsEnabled

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self._data.columns[section]
            if orientation == Qt.Vertical:
                return str(self._data.index[section])
        return None

    def sort(self, column, order):
        colname = self._data.columns[column]
        self.layoutAboutToBeChanged.emit()
        self._data.sort_values(colname, ascending=order == Qt.AscendingOrder, inplace=True)
        self.layoutChanged.emit()

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Data Analysis App'
        self.left = 100
        self.top = 100
        self.width = 800
        self.height = 600
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)

        self.table_widget = QWidget()
        self.layout = QVBoxLayout(self.table_widget)

        self.table_view = QTableView()
        self.table_view.setSortingEnabled(True)  # Enable sorting
        self.layout.addWidget(self.table_view)

        self.load_button = QPushButton('Load Data')
        self.load_button.clicked.connect(self.load_data)
        self.layout.addWidget(self.load_button)

        self.train_button = QPushButton('Train Model')
        self.train_button.clicked.connect(self.train_model)
        self.layout.addWidget(self.train_button)

        self.plot_distribution_button = QPushButton('Plot Churn Distribution')
        self.plot_distribution_button.clicked.connect(self.plot_churn_distribution)
        self.layout.addWidget(self.plot_distribution_button)

        self.plot_gender_button = QPushButton('Plot Churn by Gender')
        self.plot_gender_button.clicked.connect(self.plot_churn_by_gender)
        self.layout.addWidget(self.plot_gender_button)

        self.plot_geography_button = QPushButton('Plot Churn by Geography')
        self.plot_geography_button.clicked.connect(self.plot_churn_by_geography)
        self.layout.addWidget(self.plot_geography_button)

        self.plot_age_button = QPushButton('Plot Churn by Age Group')
        self.plot_age_button.clicked.connect(self.plot_churn_by_age_group)
        self.layout.addWidget(self.plot_age_button)

        self.plot_products_button = QPushButton('Plot Churn by Number of Products')
        self.plot_products_button.clicked.connect(self.plot_churn_by_num_products)
        self.layout.addWidget(self.plot_products_button)

        self.plot_card_button = QPushButton('Plot Churn by Credit Card')
        self.plot_card_button.clicked.connect(self.plot_churn_by_credit_card)
        self.layout.addWidget(self.plot_card_button)

        self.plot_activity_button = QPushButton('Plot Churn by Activity Status')
        self.plot_activity_button.clicked.connect(self.plot_churn_by_activity_status)
        self.layout.addWidget(self.plot_activity_button)

        self.plot_balance_button = QPushButton('Plot Churn by Balance')
        self.plot_balance_button.clicked.connect(self.plot_churn_by_balance)
        self.layout.addWidget(self.plot_balance_button)

        self.central_widget.addWidget(self.table_widget)

        # Setare stil CSS pentru interfata
        self.setStyleSheet("""
            QMainWindow {
                background-color: pink;
            }
            QPushButton {
                background-color: lightpink;
                font: bold;
            }
            QTableView {
                background-color: white;
                font: bold;
            }
        """)

    def load_data(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Load CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if fileName:
            try:
                self.data = pd.read_csv(fileName)
                self.model = PandasModel(self.data)
                self.table_view.setModel(self.model)
                self.table_view.resizeColumnsToContents()
                self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load the file: {e}")

    def train_model(self):
        try:
            # Ensure all columns are numeric, converting where necessary
            for column in self.data.columns:
                if self.data[column].dtype == 'object':
                    try:
                        self.data[column] = self.data[column].astype('float32')
                    except ValueError:
                        # If conversion fails, apply one-hot encoding
                        self.data = pd.get_dummies(self.data, columns=[column], drop_first=True)
        
            X = self.data.drop('Exited', axis=1)
            y = self.data['Exited']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            self.model = RandomForestClassifier()
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            report = classification_report(y_test, y_pred)
            QMessageBox.information(self, "Model Trained", f"Model trained successfully!\n\n{report}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Model training failed: {e}")

    def plot_churn_distribution(self):
        churn_counts = self.data['Exited'].value_counts()
        fig = px.pie(values=churn_counts, names=['Retained', 'Churned'], title='Churn Distribution')
        self.show_plot(fig)

    def plot_churn_by_gender(self):
        fig = px.histogram(self.data, x='Gender', color='Exited', barmode='group', title='Churn by Gender')
        self.show_plot(fig)

    def plot_churn_by_geography(self):
        fig = px.histogram(self.data, x='Geography', color='Exited', barmode='group', title='Churn by Geography')
        self.show_plot(fig)
    
    def plot_churn_by_age_group(self):
        try:
            self.data['AgeGroup'] = pd.cut(self.data['Age'], bins=[18, 30, 40, 50, 60, 100], labels=['18-30', '30-40', '40-50', '50-60', '60+'])
            fig = px.histogram(self.data, x='AgeGroup', color='Exited', barmode='group', title='Churn by Age Group')
            self.show_plot(fig)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Creating age groups failed: {e}")

    def plot_churn_by_num_products(self):
        fig = px.histogram(self.data, x='NumOfProducts', color='Exited', barmode='group', title='Churn by Number of Products')
        self.show_plot(fig)

    def plot_churn_by_credit_card(self):
        fig = px.histogram(self.data, x='HasCrCard', color='Exited', barmode='group', title='Churn by Credit Card')
        self.show_plot(fig)

    def plot_churn_by_activity_status(self):
        fig = px.histogram(self.data, x='IsActiveMember', color='Exited', barmode='group', title='Churn by Activity Status')
        self.show_plot(fig)

    def plot_churn_by_balance(self):
        fig = px.box(self.data, x='Exited', y='Balance', title='Churn by Balance')
        self.show_plot(fig)

    def show_plot(self, fig):
        html = fig.to_html(include_plotlyjs='cdn')
        self.plot_widget = QWebEngineView()
        self.plot_widget.setHtml(html)

        # Add return button to go back to the table view
        self.return_btn = QPushButton('Return')
        self.return_btn.clicked.connect(self.return_to_table)

        # Widget for plot
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.addWidget(self.plot_widget)
        plot_layout.addWidget(self.return_btn)

        self.central_widget.addWidget(plot_container)
        self.central_widget.setCurrentWidget(plot_container)

    def return_to_table(self):
        self.central_widget.setCurrentWidget(self.table_widget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
