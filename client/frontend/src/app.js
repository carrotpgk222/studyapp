import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

// Import each page component
import Choose from "./pages/choose/choose.js";
import Login from "./pages/login/login.js";
import Main from "./pages/main/main.js";
import Profile from "./pages/profile/profile.js";
import Register from "./pages/register/register.js";
import Review from "./pages/review/review.js";
import Survey from "./pages/survey/survey.js";

function App() {
  return (
    // 1) Wrap your entire app in BrowserRouter
    <Router>
      {/* 2) Define your routes inside <Routes> */}
      <Routes>
        <Route path="/" element={<Choose />} />
        <Route path="/login" element={<Login />} />
        <Route path="/main" element={<Main />} />
        <Route path="/profile" element={<Profile />} />
        <Route path="/register" element={<Register />} />
        <Route path="/review" element={<Review />} />
        <Route path="/survey" element={<Survey />} />

      </Routes>
    </Router>
  );
}

export default App;
