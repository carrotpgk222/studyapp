import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Choose from "./pages/choose/choose";
import Register from "./pages/register/register"; // Import your register page

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Choose />} />
        <Route path="/register" element={<Register />} />
      </Routes>
    </Router>
  );
};

export default App;