const express = require("express");
const router = express.Router();
const studyController = require("../controllers/studyController");

// Start a study session
router.post("/start", studyController.startStudySession);

// End a study session (with middlewares to calculate total time and store end time)
router.put("/end",
    studyController.endStudySession,
    studyController.calculateTotalStudyTime,
    studyController.storeEndTIme
);

// Get a study session by ID (e.g. GET /study/1)
router.get("/:id", studyController.getSessionById);

module.exports = router;
