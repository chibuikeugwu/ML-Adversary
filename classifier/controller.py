from flask import Flask, request, jsonify, make_response
from flask_cors import CORS, cross_origin

from classifier.prediction import classify

app = Flask(__name__)
CORS(app, supports_credentials=True)


# @cross_origin(origin="localhost", headers=['Content- Type', 'Authorization'])
@app.route('/classify', methods=['POST'])
@cross_origin(supports_credentials=True)
def img_classify():
    pictures = request.json
    for picture in pictures:
        img_url = picture['pictureFilePath']
        label = classify(img_url)
        picture['pictureLabel'] = label
    return pictures

if __name__ == '__main__':
    app.run(port=5000)