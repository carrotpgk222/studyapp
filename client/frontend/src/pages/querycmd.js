//=====================================================================================
// FETCH METHOD
// This function uses the fetch API to make a request to the server.
//=====================================================================================
function fetchMethod(url, callback, method = "GET", data = null, token = null) {
    console.log("fetchMethod: ", url, method, data, token);
  
    // Set headers
    const headers = {
        "Content-Type": "application/json", // Ensure proper JSON communication
    };
  
    if (token) {
        headers["Authorization"] = "Bearer " + token; // Add token if provided
    }
  
    // Prepare fetch options
    let options = {
        method: method.toUpperCase(), // HTTP method (GET, POST, etc.)
        headers: headers,
    };
  
    if (method.toUpperCase() !== "GET" && data !== null) {
        options.body = JSON.stringify(data); // Convert data to JSON string for POST/PUT
    }
    if (data !== null) {
      Object.keys(data).forEach((key) => {
          if (typeof data[key] !== "string") {
              data[key] = String(data[key]);
          }
      });
  }
    // Make the fetch call
    fetch(url, options)
      .then(response => {
        // Try to parse the response as JSON.
        return response.json()
          .catch(() => {
            // If parsing fails, return null.
            return null;
          })
          .then(parsedData => ({ status: response.status, data: parsedData }));
      })
      .then(({ status, data }) => {
        callback(status, data);
      })
      .catch(error => {
        console.error("Fetch error:", error);
        callback(500, { message: "Server error" });
      });
  }
  