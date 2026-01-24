-- Options are automatically loaded before lazy.nvim startup
-- Default options that are always set: https://github.com/LazyVim/LazyVim/blob/main/lua/lazyvim/config/options.lua
-- Add any additional options here

-- Tab and indentation settings
-- vim.opt.tabstop = 4 -- A tab character occupies 4 spaces
-- vim.opt.shiftwidth = 4 -- Indent and un-indent with 4 spaces
-- vim.opt.expandtab = true -- Use spaces instead of tabs
-- vim.opt.autoindent = true -- Copy indent from the previous line
-- vim.opt.smartindent = true -- Do smart autoindenting

-- General UI settings
vim.opt.number = true -- Show line numbers
vim.opt.relativenumber = true -- Show relative line number
vim.opt.wrap = false -- Disable line wrapping
vim.opt.scrolloff = 8 -- Keep 8 lines above/below the cursor
vim.opt.termguicolors = true -- Enable 24-bit color sadf asdas fsadfasdf sadfds asdfsd asdf sdfsadf asfasd fasf
vim.o.winborder = "rounded" -- or 'single', 'double', 'solid'

-- Persistent undo
vim.opt.undofile = true
vim.opt.undodir = os.getenv("HOME") .. "/.config/nvim/undo"

vim.g.autoformat = true -- turn on format-on-save

-- Languages
vim.g.lazyvim_cmp = "nvim-cmp"
vim.g.lazyvim_python_lsp = "basedpyright"
-- vim.keymap.set("n", "K", vim.lsp.buf.hover, { buffer = 0 })
vim.keymap.set("n", "K", function()
  vim.lsp.buf.hover({ border = "rounded", max_height = 25, max_width = 120 }) -- Change "single" to your desired border style
end, { desc = "Hover documentation" })
vim.api.nvim_set_hl(0, "FloatBorder", { fg = "#589ed7", bg = "none" })

vim.g.root_spec = {
  "lsp", -- Use the LSP server's root detection first (basedpyright will look for its config files)
  { ".git", "pyproject.toml", "pyrightconfig.json", ".env" }, -- Add specific file patterns
  "cwd", -- Fallback to the current working directory
}
