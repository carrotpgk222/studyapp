// reviewsRoutes.js
const express = require('express');
const router = express.Router();

const reviewsController = require('../controllers/reviewsController');

// POST /reviews -> create a new review
router.post('/', reviewsController.createReview);
router.post('/ai', reviewsController.callAI);
router.get("/",reviewsController.getAllReviews)

module.exports = router;
