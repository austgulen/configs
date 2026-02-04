# The target directory (Home folder)
TARGET := $(HOME)

# 'all' is the default task that runs when you just type 'make'
all:
	stow --verbose --target=$(TARGET) --restow */

# 'delete' removes all symlinks
delete:
	stow --verbose --target=$(TARGET) --delete */
