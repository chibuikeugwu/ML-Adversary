import axios from 'axios'

axios.defaults.timeout=5000;
axios.defaults.withCredentials=true;

//axios.defaults.headers.post['Content-Type']='application/x-www-from-urlencoded;charset=UTF-8'
axios.defaults.baseURL="http://localhost:8081"


export function get(url, params) {
    return new Promise((resolve, reject) => {
        axios.get(url, {params}).then(
            (response) => resolve(response.data),
            (error) => reject(error)
        );
    });
}


export function post(url, data = {}) {
    return new Promise((resolve, reject) => {
        axios.post(url, data).then(
            (response) => resolve(response.data),
            (error) => reject(error)
        );
    });
}


export function deletes(url, data = {}) {
    return new Promise((resolve, reject) => {
        axios.delete(url, data).then(
            (response) => resolve(response.data),
            (error) => reject(error)
        );
    });
}


export function put(url, data = {}) {
    return new Promise((resolve, reject) => {
        axios.put(url, data).then(
            (response) => resolve(response.data),
            (error) => reject(error)
        );
    });
}
