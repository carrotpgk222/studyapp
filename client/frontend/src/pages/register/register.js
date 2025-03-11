import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom'; // For redirection
import './register.css'; 

const Register = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: ''
  });

  // Get the navigate function for redirection
  const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    // Define the API endpoint (adjust URL/port as needed)
    const url = "http://localhost:5000/api/register"; 

    // Make a POST request with form data
    fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        username: formData.name,
        email: formData.email,
        password: formData.password
      })
    })
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then(data => {
        // Registration successful
        console.log('Registration successful:', data);
        alert('Registration successful!');
        // Redirect to login page
        navigate('/login');
      })
      .catch(error => {
        // Handle errors
        console.error('Registration error:', error);
        alert('Registration failed. Please try again.');
      });
  };

  return (
    <>
      {/* Logo placed first */}
      <div className="logo">
        <img src="/logo.png" alt="Logo" />
      </div>

      {/* Form container */}
      <div className="container">
        <div className="register-form">
          <h1>Create Your Account</h1>
          <form onSubmit={handleSubmit}>
            <label htmlFor="name">Name</label>
            <input
              type="text"
              id="name"
              name="name"
              placeholder="*Name"
              required
              value={formData.name}
              onChange={handleChange}
            />
            
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
            
            <button type="submit" className="submit-btn">Register</button>
          </form>
        </div>
      </div>
    </>
  );
};

export default Register;
