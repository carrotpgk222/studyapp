import React, { useState } from "react";
import { useNavigate } from "react-router-dom";  // For redirection
import "./login.css";

const Login = () => {
  const [formData, setFormData] = useState({ email: "", password: "" });
  const navigate = useNavigate(); // Initialize navigate for redirection

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    console.log("Logging in with:", formData);

    // Example: POST request to /login
    // Adjust the URL/port if your server is running on a different port or route
    fetch("http://localhost:5000/api/login", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        email: formData.email,
        password: formData.password
      })
    })
      .then(response => {
        // If the server responds with an error status (e.g., 401), handle it
        if (!response.ok) {
          throw new Error("Login failed");
        }
        return response.json();
      })
      .then(data => {
        console.log("Login successful:", data);

        // TODO: Possibly store tokens or user info from `data` in state/storage

        // Redirect to /survey after successful login
        navigate("/survey");
      })
      .catch(error => {
        console.error("Login error:", error);
        alert("Login failed. Please check your credentials.");
      });
  };

  return (
    <div className="login-page">
      {/* Logo (Outside the Form Container) */}
      <div className="logo">
        <img src="https://i.imgur.com/FLXaWS.png" alt="Logo" />
      </div>

      {/* Form Container */}
      <div className="container">
        <h1>Log in</h1>
        <form onSubmit={handleSubmit}>
          <label htmlFor="email">Email</label>
          <input
            type="email"
            id="email"
            name="email"
            placeholder="*Email"
            required
            value={formData.email}
            onChange={handleChange}
          />

          <label htmlFor="password">Password</label>
          <input
            type="password"
            id="password"
            name="password"
            placeholder="*Password"
            required
            value={formData.password}
            onChange={handleChange}
          />

          <div className="required-info">
            <span className="asterisk">*</span> Required Information
          </div>

          <button type="submit" className="submit-btn">
            Login
          </button>
        </form>

        <div className="login-redirect">
          <p>
            Don't have an account? <a href="/register">Register here</a>.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Login;
