#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "book.h"
#include "book_trainer.h"
#include "checkpoints.h"
#include "config.h"

// usage: ./build/glm [book1.txt book2.txt ...]
// trains a fresh model on each book in turn and saves it as the next checkpoints/N.ckpt.
// with no books it uses config::bookPath. settings for the run are in include/config.h
int main(int argc, char* argv[]) {
	std::vector<std::string> bookPaths(argv + 1, argv + argc);
	if (bookPaths.empty()) {
		bookPaths.push_back(config::bookPath);
	}

	BookTrainer trainer(nextCheckpointPath());
	std::vector<Book> seenBooks;
	seenBooks.reserve(bookPaths.size());

	for (const std::string& path : bookPaths) {
		seenBooks.push_back(loadBook(path));
		std::cout << "\n=== Training on " << seenBooks.back().name << " ===\n";
		trainer.train(seenBooks.back(), config::epochs);

		// how well it still knows every book from this run, so forgetting shows up
		std::cout << "\nValidation loss after " << seenBooks.back().name << ":\n";
		for (const Book& book : seenBooks) {
			std::cout << "  " << book.name << ": " << trainer.validationLoss(book) << '\n';
		}
	}
	return 0;
}
