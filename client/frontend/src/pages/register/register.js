import React, { useState } from 'react';
import './register.css'; 

const Register = () => {
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        password: ''
    });

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData({
            ...formData,
            [name]: value
        });
    };

    const handleSubmit = (e) => {
        e.preventDefault();

        // Make a POST request to /register (adjust the URL/port as needed)
        fetch('http://localhost:3000/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            // The server expects fields named "username", "email", and "password"
            // If your server is expecting "name" instead of "username", adjust accordingly
            body: JSON.stringify({
                username: formData.name,
                email: formData.email,
                password: formData.password
            })
        })
        .then(response => {
            if (!response.ok) {
                // Handle HTTP errors
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Handle success - for example, show a success message or redirect
            console.log('Registration successful:', data);
        })
        .catch(error => {
            // Handle any errors (network, server, etc.)
            console.error('Registration error:', error);
        });
    };

    return (
        <div className="register-container">
            <div className="logo">
                <img src="https://i.imgur.com/FLXaWSm.png" alt="Logo" />
            </div>
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
        </div>
    );
};

export default Register;
