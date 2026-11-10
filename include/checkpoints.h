#pragma once

#include <cstddef>
#include <string>

// Checkpoints live in config::checkpointFolder, one numbered file per training run:
// checkpoints/1.ckpt, checkpoints/2.ckpt, ...

// the next unused number, creating the folder if it isn't there yet
std::string nextCheckpointPath();
// the highest number so far; throws if there are no checkpoints yet
std::string latestCheckpointPath();
std::string checkpointPathFor(std::size_t number);
