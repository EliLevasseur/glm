#include "training_windows.h"

// nothing to copy out: each window is read straight from the tokens when a batch needs it
TrainingWindows buildWindows(const std::vector<std::uint32_t>& tokens) {
	return TrainingWindows{tokens};
}
