const express = require("express");
const router = express.Router();
const subjectsRoutes = require("../routes/subjectsRoutes.js");
const remindersRoutes = require("../routes/remindersRoutes.js");
const usersRoutes = require("../routes/usersRoutes.js");
const studyRoutes = require("../routes/studyRoutes.js");
const usersController = require('../controllers/usersController.js')
const reviewsRoutes = require("../routes/reviewsRoutes.js")
const bcryptMiddleware = require("../middleware/bcryptMiddleware.js")
const jwtMiddleware = require("../middleware/jwtMiddleware.js")

router.use('/subjects', subjectsRoutes);
router.use('/reminders', remindersRoutes);
router.use('/users', usersRoutes);
router.use('/study', studyRoutes);
router.use('/reviews', reviewsRoutes);
router.post("/login", usersController.login, bcryptMiddleware.comparePassword, jwtMiddleware.generateToken, jwtMiddleware.sendToken);
router.post("/register", usersController.checkUsernameOrEmailExist, bcryptMiddleware.hashPassword, usersController.register, jwtMiddleware.generateToken,jwtMiddleware.sendToken);


module.exports = router;