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
        // Handle form submission logic here (e.g., API call)
        console.log('Form Data Submitted:', formData);
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
