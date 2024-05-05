'''

Purpose: Stock price analyzer that allows users to select different time intervals to generate graphs of the stock/crypto price, fetch key information on stocks and crypo, predict the increase or decrease in a stock/cryptos price, and fetch news articles about said stock/crypto 

Input:
    - symbol: The ticker of a stock or cypto

Output:
    - data: The data gathered from the stock

'''


import os
import yfinance as yf
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QMainWindow, QLineEdit, QLabel, QApplication, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QToolBar, QFrame, QListWidget, QListWidgetItem, QMessageBox, QDialog
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import webbrowser
import sys
import requests

load_dotenv()

#class for creating a node within a doubly linked list
class HistoryNode:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

#class for managing doubly linked list for going to previoous and next pages
class HistoryManager:
    def __init__(self):
        self.head = None
        self.tail = None
        self.current = None

    def addPage(self, page):
        #creating a node with the page's data
        newNode = HistoryNode(page)
        if not self.head:
            self.head = newNode
            self.tail = newNode
            self.current = newNode
        else:
            self.current.next = None
            newNode.prev = self.current
            self.current.next = newNode
            self.current = newNode
            self.tail = newNode

            if self.current.next:
                self.current.next = None
    #traverse backwards within the doubly linked list
    def navigateBackward(self):
        if self.current.prev:
            self.current = self.current.prev
    #traverse forward within the doubly linked list
    def navigateForward(self):
        if self.current.next:
            self.current = self.current.next

#class for creating a modal to display prediction of the increase or decrease in stock price
class AnalysisModal(QDialog):
    def __init__(self, resultText, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Analysis Result")
        
        layout = QVBoxLayout(self)
        self.resultLabel = QLabel(resultText)
        layout.addWidget(self.resultLabel)


#Class for creating a stock widget showing stock's information, graphs and news
class StockInfoWidget(QWidget):
    def __init__(self, search_bar, parent=None):
        super().__init__(parent)

        self.searchBar = search_bar 
        
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setStyleSheet("background-color: #27272a; color: white;") 
                
        # Add analysis button
        self.analyzeButton = QPushButton("Analyze")
        self.analyzeButton.setStyleSheet("color: white; background-color: black; font-weight: bold;")
        self.analyzeButton.setFixedWidth(70)
        self.layout.addWidget(self.analyzeButton)
        self.analyzeButton.clicked.connect(self.analyzeStock)
                
        # Create a canvas for Matplotlib figure
        self.canvas = FigureCanvas(plt.figure())
        self.layout.addWidget(self.canvas)
                
        # Adding different time interval buttons
        self.intervalButtonsLayout = QHBoxLayout()
        self.layout.addLayout(self.intervalButtonsLayout)
                
        self.intervalButtons = []
        self.addIntervalButton("1D", self.setInterval1d)
        self.addIntervalButton("1W", self.setInterval1w)
        self.addIntervalButton("1M", self.setInterval1m)
        self.addIntervalButton("3M", self.setInterval3m)
        self.addIntervalButton("1Y", self.setInterval1y)
        self.addIntervalButton("MAX", self.setIntervalMax)
                
        # Display label for stock information
        self.infoLabel = QLabel("Key Information", self)
        self.infoLabel.setStyleSheet("color: white;")
        self.layout.addWidget(self.infoLabel)

        articles = self.getStockNews(self.searchBar.text())

        if articles:
            newsLabel = QLabel("Recent News Articles", self)
            newsLabel.setStyleSheet("color: white; font-weight: bold; font-size: 20px;")
            self.layout.addWidget(newsLabel)

            for article in articles:
                newsLink = QLabel(f"<a href=\"{article['url']}\" style=\"color: blue;\">{article['title']}</a>", self)
                newsLink.setOpenExternalLinks(True)
                self.layout.addWidget(newsLink)
            
        self.high52w = None
        self.low52w = None

        self.trainModel(self.searchBar.text())

    def openNewsArticle(self, url):
        webbrowser.open(url)
        
    def trainModel(self, symbol):
        try:
            # Retrieve historical data from yahoo_finance
            historicalData = yf.download(symbol, period='2y', interval='1d')

            if not historicalData.empty:
                historicalData['elapsedDays'] = (historicalData.index - historicalData.index.min()).days
                X = historicalData[['elapsedDays', 'Open', 'High', 'Low', 'Volume']]  
                y = historicalData['Close'] 
                XTrain, XVal, yTrain, yVal = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train the model
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)  
                self.model.fit(XTrain, yTrain)

                yPred = self.model.predict(XVal)
                print('VALIDATION MSE:', mean_squared_error(yVal, yPred))
            else:
                print(f"No historical data available for {symbol} within the specified period.")
        except Exception as e:
            print(f"Error occurred while retrieving or processing data: {e}")


    def analyzeStock(self):
        symbol = self.searchBar.text()
        historicalData = yf.download(symbol, period='2y', interval='1d')
        print('SYMBOL:', symbol)
        if not historicalData.empty:
            historicalData['elapsedDays'] = (historicalData.index - historicalData.index.min()).days

            prediction = self.model.predict(historicalData[['elapsedDays', 'Open', 'High', 'Low', 'Volume']].values)
            print('PREDICTION:', prediction)
            response = "Based on historical data from the last 2 years, it is predicted that the stock will:\n"
            if prediction[-1] > prediction[0]:
                response += "   - Increase in value."
            else:
                response += "   - Decrease in value."

            response += ' All predictions are based on historical data and may not be accurate.'
            # Display response in a modal 
            analysisModal = AnalysisModal(response)
            analysisModal.exec()
            self.getStockNews(symbol)

        else:
            QMessageBox.warning(self, "Error", f"No historical data available for {symbol} within the specified period.")


    def getStockNews(self, symbol):
        url = "https://gnews.io/api/v4/search"
        if '-' in symbol:
            symbol = symbol.split('-')[0]

        params = {
            "q": symbol, 
            "token": os.getenv("GNEWS_API_KEY"), 
            "lang": "en", 
            "country": "us",  
            "max": 5, 
        }

        # Sending GET request to the gnes api
        response = requests.get(url, params=params)
        print('RESPONSE:', response)
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            newsData = response.json()

            articles = newsData.get("articles", [])
            print('ARTICLES:', articles)
            print(f"Fetched {len(articles)} news articles for {symbol}")

            return articles

        else:
            print(f"Failed to fetch news for {symbol}. Status code: {response.status_code}")
            return []

        
    def addIntervalButton(self, text, slot):
        button = QPushButton(text)
        button.clicked.connect(slot)
        self.intervalButtons.append(button)
        self.intervalButtonsLayout.addWidget(button)

    def updateButtonColors(self, trendColor):
        for button in self.intervalButtons:
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: black;
                    color: white;
                }}
                QPushButton:hover {{
                    color: {trendColor};
                }}
            """)
            
    def setInterval1d(self):
        # Calculate start and end dates for today
        endDate = datetime.today()
        startDate = endDate - timedelta(days=1)
        
        # If the current time is after market hours
        if datetime.now().time() >= datetime.strptime("16:00:00", "%H:%M:%S").time():
            startDate -= timedelta(days=1)

        searchText = self.searchBar.text()
        self.searchedStock = searchText  
        data = yf.download(searchText, start=startDate, end=endDate)
        
        # Check if data is not empty
        if not data.empty:
            # Update the button colors based on the trend
            trendColor = 'limegreen' if data['Adj Close'].iloc[-1] > data['Adj Close'].iloc[0] else 'red'
            self.updateButtonColors(trendColor)
            # Update the graph with the new data
            self.plotStockGraph(data, searchText)
        else:
            print("No data available for the last trading day")

            
    def setInterval1w(self):
        endDate = datetime.today()
        startDate = endDate - timedelta(weeks=1)
        
        searchText = self.searchBar.text()
        self.searchedStock = searchText 
        data = yf.download(searchText, start=startDate, end=endDate)
        
        # Check if data is not empty
        if not data.empty:
            # Update the button colors based on the trend
            trendColor = 'limegreen' if data['Adj Close'].iloc[-1] > data['Adj Close'].iloc[0] else 'red'
            self.updateButtonColors(trendColor)
            # Update the graph with the new data
            self.plotStockGraph(data, searchText)
        else:
            print("No data available for the past week")
            
    def setInterval1m(self):
        endDate = datetime.today()
        startDate = endDate - timedelta(days=30)
        
        searchText = self.searchBar.text()
        self.searchedStock = searchText
        data = yf.download(searchText, start=startDate, end=endDate)
        
        # Check if data is not empty
        if not data.empty:
            # Update the button colors based on the trend
            trendColor = 'limegreen' if data['Adj Close'].iloc[-1] > data['Adj Close'].iloc[0] else 'red'
            self.updateButtonColors(trendColor)
            # Update the graph with the new data
            self.plotStockGraph(data, searchText)
        else:
            print("No data available for the past month")

            
    def setInterval3m(self):
        endDate = datetime.today()
        startDate = endDate - timedelta(days=90)
        
        searchText = self.searchBar.text()
        self.searchedStock = searchText
        data = yf.download(searchText, start=startDate, end=endDate)
        
        # Check if data is not empty
        if not data.empty:
            # Update the button colors based on the trend
            trendColor = 'limegreen' if data['Adj Close'].iloc[-1] > data['Adj Close'].iloc[0] else 'red'
            self.updateButtonColors(trendColor)
            # Update the graph with the new data
            self.plotStockGraph(data, searchText)
        else:
            print("No data available for the past 3 months")

            
    def setInterval1y(self):
        endDate = datetime.today()
        startDate = endDate - timedelta(days=365)
        
        searchText = self.searchBar.text()
        self.searchedStock = searchText  
        data = yf.download(searchText, start=startDate, end=endDate)
        
        # Check if data is not empty
        if not data.empty:
            # Update the button colors based on the trend
            trendColor = 'limegreen' if data['Adj Close'].iloc[-1] > data['Adj Close'].iloc[0] else 'red'
            self.updateButtonColors(trendColor)
            # Update the graph with the new data
            self.plotStockGraph(data, searchText)
        else:
            print("No data available for the past year")
            
    def setIntervalMax(self):
        endDate = datetime.today()
        startDate = endDate - timedelta(days=2 * 365)
        
        searchText = self.searchBar.text()
        self.searchedStock = searchText 
        data = yf.download(searchText, start=startDate, end=endDate)
        
        # Check if data is not empty
        if not data.empty:
            # Calculate and store 52-week high and low values
            self.high52w = data['High'].rolling(window=365).max().iloc[-1]
            self.low52w = data['Low'].rolling(window=365).min().iloc[-1]
            
            # Update the button colors based on the trend
            trendColor = 'limegreen' if data['Adj Close'].iloc[-1] > data['Adj Close'].iloc[0] else 'red'
            self.updateButtonColors(trendColor)
            # Update the graph with the new data
            self.plotStockGraph(data, searchText)
        else:
            print("No data available for the past 2 years")

            
    def setStockInfo(self, stockInfo):
        self.infoLabel.setText(stockInfo)
        
    def plotStockGraph(self, data, symbol):
        if data.empty:
            QMessageBox.warning(self, 'Error', 'No data available for this stock/cryptocurrency')
            return

        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)  

        # Graph the closing prices
        data.index = pd.to_datetime(data.index)
        upperSymbol = symbol.upper()
        ax.plot(data.index, data['Adj Close'], label=f'{upperSymbol} Adj Close Price', color='black')

        # Find the trend
        if data['Adj Close'].iloc[-1] > data['Adj Close'].iloc[0]:
            trendColor = 'limegreen'
        else:
            trendColor = 'red'

        # Graph the trend line
        ax.plot(data.index, data['Adj Close'], color=trendColor, linestyle='-')
        companyName = yf.Ticker(symbol).info['longName']
        ax.set_title(f'{companyName} Adj Close Price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.grid(True)
        ax.legend()
        self.updateButtonColors(trendColor)
        
        self.displayStockInfo(data)
        self.canvas.draw()

    def displayStockInfo(self, data):
        todayData = data[data.index.normalize() == pd.Timestamp(datetime.today().date())]
        print('TODAY DATA:', todayData)

        # If today's data is not available, get the most recent data
        if todayData.empty:
            todayData = data.iloc[-1:]

        # Calculate today's high and low
        todayHigh = todayData['High'].max() if not todayData.empty else None
        todayLow = todayData['Low'].min() if not todayData.empty else None

        # Calculate the date 52 weeks ago
        startDate52w = datetime.today() - timedelta(weeks=52)

        data52w = data[data.index >= startDate52w]

        # Get the 52-week High and Low values
        high52w = data52w['High'].max()
        low52w = data52w['Low'].min()

        # Format money values with $ sign and two decimal places
        stockInfoFormatted = {
            'Today\'s High': f'${todayHigh:.2f}' if todayHigh is not None else 'N/A',
            'Today\'s Low': f'${todayLow:.2f}' if todayLow is not None else 'N/A',
            '52 Week High': f'${high52w:.2f}',
            '52 Week Low': f'${low52w:.2f}',
            'Volume': f'{data.iloc[-1]["Volume"]:,}'
        }

        stockInfoStr = ''.join([f"<span style='border-bottom:1px solid gray;'><b><font color='white' style='font-size:17px;'>{k}</font></b></span><br>{v}<br><br>" for k, v in stockInfoFormatted.items()])        
        self.infoLabel.setText(stockInfoStr)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Stock Analyzer Project")
        self.setGeometry(200, 200, 1200, 1000)
        self.setStyleSheet("background-color: #27272a; color: white;")  

        containerWidget = QWidget(self)
        self.setCentralWidget(containerWidget)

        # Layout for container
        self.containerLayout = QVBoxLayout(containerWidget)

        # Creating a toolbar
        self.toolbar = QToolBar(self)
        self.addToolBar(self.toolbar)

        # Creating the back button
        self.backButton = QPushButton(self)
        self.backButton.setIcon(QIcon('./arrow-left-solid.svg')) 
        self.backButton.setStyleSheet("background-color: white; color: gray;")
        self.backButton.clicked.connect(self.back)
        self.backButton.setFixedWidth(40)

        # Creating the forward button
        self.forwardButton = QPushButton(self)
        self.forwardButton.setIcon(QIcon('./arrow-right-solid.svg'))  
        self.forwardButton.setStyleSheet("background-color: white; color: gray;")
        self.forwardButton.clicked.connect(self.forward)
        self.forwardButton.setFixedWidth(40)

        # Add the buttons to the toolbar
        self.toolbar.addWidget(self.backButton)
        self.toolbar.addWidget(self.forwardButton)

        # Creating the search bar
        self.searchBar = QLineEdit()
        self.searchBar.setFixedWidth(280)  
        self.searchBar.returnPressed.connect(self.handleSearch)  
        self.searchBar.setPlaceholderText("Search for a stock(ticker)...")

        self.homeLabel = QLabel("WeTrade", self)
        self.homeLabel.setStyleSheet("color: white; font-size: 25px; font-weight: bold;")
        self.topLayout = QHBoxLayout()

        self.topLayout.addStretch()
        self.topLayout.addWidget(self.searchBar)
        self.topLayout.addStretch()

        self.containerLayout.addWidget(self.homeLabel)
        self.containerLayout.addLayout(self.topLayout)

        self.div1 = QFrame()
        self.div2 = QFrame()

        self.div1.setStyleSheet("background-color: black;")
        self.div2.setStyleSheet("background-color: black;")

        self.div1.setFixedSize(300, 800)
        self.div2.setFixedSize(300, 800)

        divLayout = QHBoxLayout()

        divLayout.addWidget(self.div1)
        divLayout.addWidget(self.div2)

        self.containerLayout.addLayout(divLayout)

        divLayout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
        listWidget1 = QListWidget(self.div1)
        listWidget2 = QListWidget(self.div2)

        listWidget1.setStyleSheet("font-size: 18px; font-weight: 600;")
        listWidget2.setStyleSheet("font-size: 18px; font-weight: 600;")

        title1 = QLabel("Popular Stocks")
        title2 = QLabel("Popular Crypto")
        title1.setStyleSheet("font-size: 24px; font-weight: bold; color:") 
        title2.setStyleSheet("font-size: 24px; font-weight: bold;")

        # Creating a list of popular stocks
        popularStocks = ['AMZN', 'AAPL', 'NVDA', 'MSFT', 'TSLA', 'META','GOOGL', 'AMD', 'PYPL', 'NFLX', 'F', 'DIS', 'SBUX', 'GPRO']
        self.updateListItemsWithPrices(listWidget1, popularStocks)

        # Creating a list of popular cryptocurrencies
        popularCryptos = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'USDC-USD', 'DOGE-USD', 'SHIB-USD', 'AVAX-USD', 'BCH-USD', 'LINK-USD', 'ETC-USD', 'UNI-USD','XLM-USD', 'ATOM-USD', 'XTZ-USD']
        self.updateListItemsWithPrices(listWidget2, popularCryptos)

        container1 = QWidget()
        container2 = QWidget()

        container1.setStyleSheet("border: 1px solid gray;")
        container2.setStyleSheet("border: 1px solid gray;")

        layout1 = QVBoxLayout()
        layout2 = QVBoxLayout()

        container1.setLayout(layout1)
        container2.setLayout(layout2)

        layout1.addWidget(title1)
        layout2.addWidget(title2)

        layout1.addWidget(listWidget1)
        layout2.addWidget(listWidget2)

        div1Layout = QVBoxLayout(self.div1)
        div2Layout = QVBoxLayout(self.div2)

        div1Layout.addWidget(container1)
        div2Layout.addWidget(container2)

        self.containerLayout.addStretch()

        self.stockInfoWidget = StockInfoWidget(self.searchBar) 
        self.searchBar.setStyleSheet(
            """
            QLineEdit {
                background-color: black;
                color: white;
                border: 1px solid gray;
                border-radius: 5px;
                padding: 5px;
            }
            """
        )

        self.historyManager = HistoryManager()

        self.initializeHomePage()


    def fetchCurrentPrices(self, symbols):
        currentPrices = {}
        openingPrices = {}
        for symbol in symbols:
            data = yf.Ticker(symbol)
            history = data.history(period="2d")
            currentPrice = history["Close"].iloc[-1]
            openingPrice = history["Close"].iloc[-2]  
            currentPrices[symbol] = currentPrice
            openingPrices[symbol] = openingPrice
            # print(f"{symbol} - ${currentPrice:.2f}")
        return currentPrices, openingPrices

    def fetchCurrentPrices(self, symbols):
        currentPrices = {}
        openingPrices = {}
        for symbol in symbols:
            # Fetch the current price using yfinance
            data = yf.Ticker(symbol)
            history = data.history(period="1d")
            currentPrice = history["Close"].iloc[-1]
            openingPrice = history["Open"].iloc[0]
            currentPrices[symbol] = currentPrice
            openingPrices[symbol] = openingPrice
            # print(f"{symbol} - ${currentPrice:.2f}")
        return currentPrices, openingPrices


    def updateListItemsWithPrices(self, listWidget, symbols):
        # Fetch current prices and opening prices for the symbols
        currentPrices, openingPrices = self.fetchCurrentPrices(symbols)

        for symbol in symbols:
            if symbol in currentPrices:
                price = currentPrices[symbol]
                color = "limegreen" if price >= openingPrices[symbol] else "red"
                button = QPushButton(f"{symbol}     ${price:.2f}")
                button.setStyleSheet(f"color: {color};")
                button.clicked.connect(lambda _, s=symbol: self.setSearchAndHandle(s))
                item = QListWidgetItem(listWidget)
                listWidget.setItemWidget(item, button)

        listWidget.setSpacing(10)

    def setSearchAndHandle(self, symbol):
        self.searchBar.setText(symbol)
        self.handleSearch()

    def initializeHomePage(self):
        self.historyManager.addPage(self.homeLabel)
        if self.stockInfoWidget is not None:
            self.containerLayout.removeWidget(self.stockInfoWidget)
            self.stockInfoWidget.setVisible(False)
        self.homeLabel.setVisible(True)
        self.containerLayout.update()
        self.containerLayout.activate()


    def handleSearch(self):
        # Hide the home label when a search is performed
        self.homeLabel.setVisible(False)
        self.div1.setVisible(False)
        self.div2.setVisible(False)
        searchText = self.searchBar.text()

        # Retrieve stock data for the entered symbol
        currentDate = datetime.today()
        print('Current Date:', currentDate)

        try:
            data = yf.download(searchText, start="2022-01-01", end=currentDate)
        except KeyError:
            QMessageBox.warning(self, 'Error', 'Stock/Cryptocurrency not found')
            return

        print(data.columns)
        stockInfo = data.head().to_string()

        newStockInfoWidget = StockInfoWidget(self.searchBar)

        newStockInfoWidget.setStockInfo(stockInfo)
        newStockInfoWidget.plotStockGraph(data, searchText)

        newStockInfoWidget.searchedStock = searchText

        self.historyManager.addPage(newStockInfoWidget)

        # Hide the current stock info widget
        if self.stockInfoWidget is not None:
            self.stockInfoWidget.setVisible(False)

        # Set the new stock info widget as the current one
        self.stockInfoWidget = newStockInfoWidget

        # Add the new stock info widget to the container layout
        self.containerLayout.addWidget(self.stockInfoWidget)


    def back(self):
        self.historyManager.navigateBackward()
        previousWidget = self.historyManager.current.data
        if self.stockInfoWidget is not None:
            self.stockInfoWidget.setVisible(False)
        self.stockInfoWidget = previousWidget
        self.stockInfoWidget.setVisible(True)
        self.homeLabel.setVisible(self.historyManager.current.data is self.homeLabel)
        self.div1.setVisible(self.historyManager.current.data is self.homeLabel)
        self.div2.setVisible(self.historyManager.current.data is self.homeLabel)

    def forward(self):
        self.historyManager.navigateForward()
        
        nextWidget = self.historyManager.current.data
        
        if self.stockInfoWidget is not None:
            self.stockInfoWidget.setVisible(False)
            
        self.stockInfoWidget = nextWidget
        
        self.stockInfoWidget.setVisible(True)
        
        
        self.div1.setVisible(self.historyManager.current.data is self.homeLabel)
        self.div2.setVisible(self.historyManager.current.data is self.homeLabel)
        self.homeLabel.setVisible(self.historyManager.current.data is self.homeLabel)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
