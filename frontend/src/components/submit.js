import React, { useState, useEffect, useRef } from 'react';
import Button from '@mui/material/Button';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import SendIcon from '@mui/icons-material/Send';
import { Chart } from 'chart.js';
// import { LoadingButton } from '@mui/lab';

const button_theme = createTheme({
  palette: { primary: { main: '#00FFFF' }, text: { primary: '#ffffff' } },
});

let stockSymbol = "";

function App() {
  const [prediction, setPrediction] = useState({});
  const [stockDates, setStockDates] = useState([]);
  const [stockClosePrices, setStockClosePrices] = useState([]);
  const [stockOpenPrices, setStockOpenPrices] = useState([]);
  const [predictionSet, setPredictionSet] = useState(false);

  const chartRef = useRef(null);
  const chartInstanceRef = useRef(null);

  const handleBeforeSubmit = async () => {
    setPredictionSet(true);
    stockSymbol = "TSLA";

    let response = null;

    try {
      response = await fetch('http://localhost:5000/prepredict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ stock_symbol: stockSymbol })
      });

      const data = await response.json();

      setPrediction(data);
      setStockDates(data.dates);
      setStockClosePrices(data.pricescl);
      setStockOpenPrices(data.pricesop);
    } 
    
    catch {
      console.log("error");
    }
  };

  const handleSubmit = async (event) => {
    setPredictionSet(true);
    event.preventDefault();
    stockSymbol = event.target.elements.stockSymbol.value;

    let response = null;

    try {
      response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ stock_symbol: stockSymbol })
      });

      const data = await response.json();

      setPrediction(data);
      setStockDates(data.dates);
      setStockClosePrices(data.pricescl);
      setStockOpenPrices(data.pricesop);
    } 
    
    catch {
      console.log("error");
    }
  };

  useEffect(() => {
    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
    }
  
    if (stockDates.length > 0 && stockClosePrices.length > 0) {
      const chartCanvas = document.getElementById('canvas');
      chartInstanceRef.current = new Chart(chartCanvas, {
        type: 'line',
        data: {
          labels: stockDates,
          datasets: [
            {
              label: stockSymbol + ' Price',
              data: stockClosePrices,
              borderColor: 'rgb(0,255,255)',
              backgroundColor: 'rgba(0,0,0,0)',
              borderWidth: 2,
              fill: false,
              pointRadius: 0,
            },
          ],
        },
        options: {
          scales: {
            x: {
              ticks: {
                maxRotation: 0,
                minRotation: 0,
                backgroundColor: 'red'
              },
            },
          },
        },
      });
  
      const chartContainer = document.getElementById('graph');
      chartContainer.append(chartInstanceRef.current.canvas);
    }
  }, [stockDates, stockClosePrices]);
  
  let currTime = new Date().toLocaleTimeString('en-US', {
    timeZone: 'America/New_York', hour12: false, timeZoneName: 'short',
    }).replace(/:/g, '').split(" ")[0];
  
  if(parseInt(currTime) < 163000 && parseInt(currTime) > 93000){
    document.getElementById("live").innerHTML = '<div class = "circle" style = "background-color: green;"></div><div class = "text" style = "color: green;">MARKET LIVE</div>';
  }
  else{
    document.getElementById("live").innerHTML = '<div class = "circle" style = "background-color: red;"></div><div class = "text" style = "color: red;">AFTER HOURS</div>';
  }

  if(prediction.next_day_open != null){
    var tdyOpen, tdyClose, predOpen, predClose, innerTextOp, innerTextCl, innerPredOp, innerPredCl;

    if(prediction.pricesop[prediction.pricesop.length - 1] > prediction.pricescl[prediction.pricescl.length - 2]){
      tdyOpen = '<div class = "green-arrow"></div> <div id = "textJSOp" style = "color: white; font-weight: bolder;"></div>';
      innerTextOp = "$" + prediction.pricesop[prediction.pricesop.length - 1].toFixed(2);
    }
    else{
      tdyOpen = '<div class = "red-arrow"></div> <div id = "textJSOp" style = "color: white; text-shadow: #fff; font-weight: bolder;"></div>';
      innerTextOp = "$" + prediction.pricesop[prediction.pricesop.length - 1].toFixed(2);
    }

    if(prediction.pricescl[prediction.pricescl.length - 1] > prediction.pricesop[prediction.pricesop.length - 1]){
      tdyClose = '<div class = "green-arrow"></div> <div id = "textJSCl" style = "color: white;font-weight: bolder;"></div>';
      innerTextCl = "$" + prediction.pricescl[prediction.pricescl.length - 1].toFixed(2);
    }
    else{
      tdyClose = '<div class = "red-arrow"></div> <div id = "textJSCl" style = "color: white; font-weight: bolder;"></div>';
      innerTextCl = "$" + prediction.pricescl[prediction.pricescl.length - 1].toFixed(2);
    }

    if(prediction.next_day_open > prediction.pricescl[prediction.pricescl.length - 1]){
      predOpen = '<div class = "green-arrow"></div> <div id = "textPredOp" style = "color: white; font-weight: bolder;"></div>';
      innerPredOp = "$" + prediction.next_day_open;
    }
    else{
      predOpen = '<div class = "red-arrow"></div> <div id = "textPredOp" style = "color: white; font-weight: bolder;"></div>';
      innerPredOp = "$" + prediction.next_day_open;
    }

    if(prediction.next_day_close > prediction.next_day_open){
      predClose = '<div class = "green-arrow"></div> <div id = "textPredCl" style = "color: white; font-weight: bolder;"></div>';
      innerPredCl = "$" + prediction.next_day_close;
    }
    else{
      predClose = '<div class = "red-arrow"></div> <div id = "textPredCl" style = "color: white; font-weight: bolder;"></div>';
      innerPredCl = "$" + prediction.next_day_close;
    }

    document.getElementById("buy-or-not").innerHTML = prediction.next_day_close
     > prediction.next_day_open ? "BUY TOMORROW" : "DON'T BUY TOMORROW";
    

    document.getElementById("market-cap").innerText = prediction.market_cap != null ? format(prediction.market_cap) : null;
    document.getElementById("high").innerText = "$" + prediction.high;
    document.getElementById("low").innerText = "$" + prediction.low;
    document.getElementById("volume").innerText = prediction.volume != null ? format(prediction.volume) : null;
    document.getElementById("eps").innerText = "$" + prediction.eps;
    document.getElementById("dividend-yield").innerText = prediction.dividend_yield == "N/A" ? prediction.dividend_yield : prediction.dividend_yield + "%";
    document.getElementById("pe-ratio").innerText = prediction.pe_ratio;
    document.getElementById("volatility").innerText = prediction.volatility + "% ";
    
    document.getElementById("tdy-open").innerHTML = tdyOpen;
    document.getElementById("tdy-close").innerHTML = tdyClose;
    document.getElementById("pred-open").innerHTML = predOpen;
    document.getElementById("pred-close").innerHTML = predClose;

    document.getElementById("textJSOp").innerText = innerTextOp;
    document.getElementById("textJSCl").innerText = innerTextCl;
    document.getElementById("textPredOp").innerText = innerPredOp;
    document.getElementById("textPredCl").innerText = innerPredCl;
  }

  if(predictionSet){
    return (
      <div className="ret">
        <div className="form">
          <form onSubmit={handleSubmit}>
            <div className="input-container">
              <input className="input" type="text" name="stockSymbol" placeholder="Enter Stock/Crypto Symbol" required />
              <ThemeProvider theme={button_theme}>
                <Button className="button" variant="contained" type="submit"> <SendIcon /> </Button>
              </ThemeProvider>
            </div>
          </form>
        </div>
      </div>
    );
  }

  else{
    handleBeforeSubmit();
    return (
      <div className="ret">
        <div className="form">
          <form onSubmit={handleSubmit}>
            <div className="input-container">
              <input className="input" type="text" name="stockSymbol" placeholder="Enter Stock/Crypto Symbol" required />
              <ThemeProvider theme={button_theme}>
                <Button className="button" variant="contained" type="submit"> <SendIcon /> </Button>
              </ThemeProvider>
            </div>
          </form>
        </div>
      </div>
    );
  }
}

function format(number){
  const suffixes = ['', 'THOUSAND', 'MILLION', 'BILLION', 'TRILLION'];
  let suffixIndex = 0;
  let formattedNumber = number;

  while (formattedNumber >= 1000) {
    formattedNumber /= 1000;
    suffixIndex++;
  }

  return `${formattedNumber.toFixed(2)} ${suffixes[suffixIndex]}`;
}

export default App;
