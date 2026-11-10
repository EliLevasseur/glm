#pragma once

#include <string>

#include "batch.h"
#include "training_windows.h"

// One book, read, split and windowed once so it can be trained on and checked later.
struct Book {
	std::string name;
	TrainingWindows trainWindows;
	TrainingWindows validationWindows;   // the end of the book, after trainingFraction
	Batch validationBatch;
};

Book loadBook(const std::string& path);
