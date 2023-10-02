import axios from "axios";

export function postToClassifier(data) {
    return new Promise((resolve, reject) => {
        axios.post('http://localhost:5000/classify', data,  {
            headers: {
                'Content-Type': 'application/json'
            },
            withCredentials: true,
            timeout: 0}).then(
            (response) => resolve(response.data),
            (error) => reject(error)
        );
    });
}
