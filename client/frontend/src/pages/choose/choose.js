import React from "react";
import { useNavigate } from "react-router-dom";  // Import useNavigate from react-router-dom
import "./choose.css"; // Make sure to create and import your CSS file

const Choose = () => {
  const navigate = useNavigate(); // Initialize the navigate function

  // Navigate to "/login"
  const handleLogin = () => {
    navigate("/login");
  };

  // Navigate to "/register"
  const handleSignup = () => {
    navigate("/register");
  };

  return (
    <div className="container">
      <div className="logo">
        <img src="https://i.imgur.com/FLXaWSm.png" alt="App Logo" />
      </div>
      <h1 className="title">
        <span className="highlight">Unlock</span> Your <span className="highlight">Potential</span>
      </h1>

      <p className="descriptionOne">
        Manage your time, improve your education, and get responsive feedback.
      </p>
      <p className="descriptionTwo">
        Customize your learning journey today!
      </p>

      <p className="slogan">Deeply Seeking Knowledge, One Step at a Time</p>

      <div className="button-container">
        {/* Login Button */}
        <button className="login-button" onClick={handleLogin} id="login-btn">
          Log in
        </button>
        
        {/* Sign Up Button */}
        <button className="signup-button" onClick={handleSignup} id="signup-btn">
          Sign up
        </button>
      </div>
    </div>
  );
};

export default Choose;
