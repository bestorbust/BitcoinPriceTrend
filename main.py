from flask import Flask, render_template, request
from gru_predict import get_gru_trend_for_date

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    trend_result = None
    if request.method == 'POST':
        input_date = request.form['date']
        trend_result = get_gru_trend_for_date(input_date)
    return render_template('index.html', trend=trend_result)

if __name__ == '__main__':
    app.run(debug=True)
