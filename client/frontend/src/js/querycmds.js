import axios from "axios";

function axiosMethod(url, callback, method = "GET", data = null, token = null) {
  console.log("axiosMethod ", url, method, data, token);

  const headers = {};

  if (data) {
    headers["Content-Type"] = "application/json";
  }

  if (token) {
    headers["Authorization"] = "Bearer " + token;
  }

  const axiosConfig = {
    method: method.toUpperCase(),
    url: url,
    headers: headers,
    data: data,
  };

  axios(axiosConfig)
    .then((response) => callback(response.status, response.data))
    .catch((error) => console.error(`Error from ${method} ${url}:`, error));
}

export default axiosMethod;