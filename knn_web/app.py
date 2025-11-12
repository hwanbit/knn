from flask import Flask, render_template, request, send_file
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import io
import base64
import numpy as np
import os

# 한글 폰트 설정 (그래프용)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Flask 앱 생성
app = Flask(__name__)

# 저장된 모델 불러오기
with open('./model/knn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 붓꽃 데이터 및 이름
iris = load_iris()
iris_names = iris.target_names
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['label'] = iris_names[iris.target]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    img_url = None
    plot_url = None

    if request.method == 'POST':
        try:
            # 입력값 받기
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])

            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            pred = model.predict(features)[0]
            prediction = iris_names[pred]

            # 품종별 이미지 경로 지정
            img_map = {
                'setosa': '/static/setosa.jpeg',
                'versicolor': '/static/versicolor.jpg',
                'virginica': '/static/virginica.jpg'
            }
            img_url = img_map.get(prediction, None)

            # KNN 시각화 (petal length / width)
            plt.figure(figsize=(5, 4))
            sns.scatterplot(
                x=df['petal length (cm)'], y=df['petal width (cm)'],
                hue=df['label'], palette='Set2', s=60
            )
            plt.scatter(petal_length, petal_width, c='red', s=120, marker='x', label='input data')
            plt.legend()
            plt.title('KNN Result Visualization')
            plt.xlabel('Petal Length (cm)')
            plt.ylabel('Petal Width (cm)')

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()
            plt.close()

        except Exception as e:
            prediction = f"입력 오류: {e}"

    return render_template('index.html',
                           prediction=prediction,
                           img_url=img_url,
                           plot_url=plot_url)

# 모델 다운로드 라우트
@app.route('/download_model')
def download_model():
    model_path = './model/knn_model.pkl'
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    else:
        return "모델 파일이 존재하지 않습니다.", 404

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)