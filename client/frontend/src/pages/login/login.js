import { useState } from "react";
import "./login.css"; 

const Login = () => {
  const [formData, setFormData] = useState({ email: "", password: "" });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("Logging in with:", formData);
  };

  return (
    <div className="login-page">
      {/* Logo (Outside the Form Container) */}
      <div className="logo">
        <img src="https://i.imgur.com/FLXaWSm.png" alt="Logo" />
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
