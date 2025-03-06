import React from "react";
import "./choose.css"; // Make sure to create and import your CSS file

const Choose = () => {
  // Placeholder functions for navigation (Update when backend is ready)
  const handleLogin = () => {
    console.log("Login button clicked");
    // TODO: Implement navigation or authentication logic here
  };

  const handleSignup = () => {
    console.log("Sign up button clicked");
    // TODO: Implement navigation or authentication logic here
  };

  return (
    <div className="container">
      <div className="logo">
        <img src="https://i.imgur.com/FLXaWSm.png" alt="App Logo" />
      </div>
      <h1 className="title">
        <span className="highlight">Unlock</span> Your <span className="highlight">Potential</span>
      </h1>

      <p className="descriptionOne">Manage your time, improve your education, and get responsive feedback.</p>
      <p className="descriptionTwo">Customize your learning journey today!</p>

      <p className="slogan">Deeply Seeking Knowledge, One Step at a Time</p>

      <div className="button-container">
        {/* Login Button: Update `onClick` when backend is ready */}
        <button className="login-button" onClick={handleLogin} id="login-btn">
          Log in
        </button>
        
        {/* Sign Up Button: Update `onClick` when backend is ready */}
        <button className="signup-button" onClick={handleSignup} id="signup-btn">
          Sign up
        </button>
      </div>
    </div>
  );
};

export default Choose;

