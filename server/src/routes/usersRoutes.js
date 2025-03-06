const express = require("express");
const router = express.Router();
const usersController = require("../controllers/usersController");


router.get("/",usersController.getAllUsers)
router.get("/:user_id", usersController.getUserByUserId)
module.exports = router;