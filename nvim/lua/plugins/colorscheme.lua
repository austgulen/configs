return {
  -- import themes
  {
    "nyoom-engineering/oxocarbon.nvim",
    -- Add in any other configuration;
    --   event = foo,
    --   config = bar
    --   end,
  },
  {
    { "iagorrr/noctishc.nvim" },
  },
  { "miikanissi/modus-themes.nvim", priority = 1000 },
  { "bluz71/vim-moonfly-colors", name = "moonfly", lazy = false, priority = 1000 },
  { "Shatur/neovim-ayu" },
  -- {
  --   "jpwol/thorn.nvim",
  --   lazy = false,
  --   priority = 1000,
  --   -- name = "thorn",
  --   opts = {
  --     theme = "dark", -- 'light' or 'dark' - defaults to vim.o.background if unset
  --     background = "warm", -- options are 'warm' and 'cold'
  --     styles = {
  --       keywords = { italics = false, bold = true },
  --       comments = { italics = true, bold = false },
  --       strings = { italics = true, bold = false },
  --
  --       diagnostic = {
  --         underline = true, -- if true, flat underlines will be used. Otherwise, undercurls will be used
  --
  --         -- true will apply the bg highlight, false applies the fg highlight
  --         error = { highlight = true },
  --         hint = { highlight = false },
  --         info = { highlight = false },
  --         warn = { highlight = false },
  --       },
  --     },
  --
  --     on_highlights = function(hl, c)
  --       -- hl.LspInlayHint = { fg = c.fg, italic = true }
  --       hl.LspInlayHint = { fg = c.gray, bg = "#1D282F", italic = true }
  --       -- hl.LspInlayHint = { bg = c.bg, fg = c.fg }
  --     end,
  --
  --     -- on_highlights = function(hl, palette) end, -- apply your own highlights
  --   },
  -- },
  {
    "jpwol/thorn.nvim",
    lazy = false,
    priority = 1000,
    opts = {},
  },

  {
    "tahayvr/matteblack.nvim",
    lazy = false,
    priority = 1000,
    opts = {},
    config = function()
      --   require("matteblack").colorscheme()
      --   vim.cmd.colorscheme("matteblack")
    end,
    -- require("matteblack").colorscheme(),
  },
  {
    "EdenEast/nightfox.nvim",
    lazy = false,
    priority = 1000,
    opts = { style = "terafox" },
    -- config = function() end,
  },
  {
    "vague-theme/vague.nvim",
    lazy = false,
    priority = 1000,
    opts = {},
  },
  {
    "ficcdaf/ashen.nvim",
    -- optional but recommended,
    -- pin to the latest stable release:
    -- tag = "*",
    lazy = false,
    priority = 1000,
    -- configuration is optional!
    opts = {
      -- your settings here
    },
  },

  {
    "scottmckendry/cyberdream.nvim",
    lazy = false,
    priority = 1000,
    opts = {
      transparent = true,
      italic_comments = true,
      hide_fillchars = false,
      borderless_pickers = false,
      borderless_telescope = true,
      terminal_colors = true,
    },
    -- config = function()
    --   require("bluloco").setup({
    --     italic_comments = true,
    --     terminal = vim.fn.has("gui_running") == 1, -- bluoco colors are enabled in gui terminals per default.
    --     guicursor = true,
    --     rainbow_headings = false, -- if you want different colored headings for each heading level
    --   })
    --   -- your optional config goes here, see below.
    -- end,
  },
  {
    "alljokecake/naysayer-theme.nvim",
    name = "naysayer", -- Optional, but good practice
  },
  {
    "craftzdog/solarized-osaka.nvim",
    opts = {
      -- transparent = flase,
      styles = {
        comments = { italic = true }, -- Italics for comments
        keywords = { italic = true }, -- Italics for keywords
        functions = {}, -- No special style for functions
        variables = {}, -- No special style for variables
      },
      telescope = { enabled = true }, -- Enable Telescope integration
      navic = { enabled = true }, -- Enable Navic (line numbers/status)
      lsp = { enabled = true }, -- LSP semantic tokens
      treesitter = { enabled = true }, -- Tree-sitter highlighting
      transparent = false, -- Set to true for no background
      dim_inactive = false, -- Dim inactive windows
      lualine_bols = true,

      on_highlights = function(hl, c) -- Optional: Custom highlight overrides
        hl["@field"] = { fg = c.green }
        hl["@property"] = { fg = c.blue }
      end,
    },
  },
  {
    "rose-pine/neovim",
    name = "rose-pine",
    -- config = function()
    -- 	vim.cmd("colorscheme rose-pine")
    -- end
  },
  { "miikanissi/modus-themes.nvim", priority = 1000 },
  {
    "LazyVim/LazyVim",
    opts = {
      -- colorscheme = "naysayer",
      -- colorscheme = "matteblack",
      -- colorscheme = "bluloco",
      -- colorscheme = "solarized-osaka",
      -- colorscheme = "cyberdream",
      -- colorscheme = "monokai-pro",
      -- colorscheme = "moonfly",
      -- colorscheme = "rose-pine-moon",
      -- colorscheme = "terafox",
      colorscheme = "thorn",
      -- colorscheme = "carbonfox",
    },
  },
  -- {
  --   "loctvl842/monokai-pro.nvim",
  --   lazy = false,
  --   priority = 1000,
  --   opts = {
  --     -- transparent_background = true,
  --     filter = "classic",
  --   },
  -- },
  -- vim.api.nvim_set_hl(0, "Comment", { fg = "#e5c07b", italic = true }),
}

-- local M = {}
--
-- function M.setup(opts)
--   if opts.theme == "dark" then
--     -- stylua: ignore
--     return {
--       bg         = opts.background == "warm" and "#152326" or "#1D282F",
--       fg         = "#DBD0C6",
--
--       number     = opts.background == "warm" and "#234847" or "#314654",
--
--       white      = "#D9D3CE",
--       gray       = "#91A4AD",
--       green      = "#568270",
--       green0     = "#6FA791",
--       green1     = "#9EBB9C",
--       yellow     = "#FFD7AA",
--       yellow0    = "#F7B982",
--       orange     = "#F9ADA0",
--       blue       = "#86BFD0",
--       blue0      = "#A7CBEA",
--       lightblue  = "#9FCFC3",
--       lightgreen = "#96C2A1",
--       pink       = "#D9ADD4",
--       cyan       = "#79C2B6",
--       cyan0      = "#87CBB1",
--       red        = "#D2696C",
--       red0       = "#FA5056",
--       red1       = "#E89396",
--
--       cursorline = opts.background == "warm" and "#1D3034" or "#21313B",
--       separator  = "#0B1213",
--
--       statusbar  = opts.background == "warm" and "#111F22" or "#152128",
--       status_sep = opts.background == "warm" and "#203336" or "#1A2C37",
--
--       bg_float   = opts.background == "warm" and "#0F1A1C" or "#1A2328",
--
--       bg_visual  = opts.background == "warm" and "#38524F" or "#223B49",
--
--       diff = {
--         add = "#435B55",
--         change = "#23363B",
--         delete = "#704C4E",
--       },
--
--       git = {
--         add = "#94C68B",
--         change = "#6DAAE3",
--         delete = "#D8464B",
--       },
--
--       none = "NONE",
--     }
--   elseif opts.theme == "light" then
--     -- stylua: ignore
--     return {
--       bg         = opts.background == "warm" and "#CDE3C8" or "#C9E4D4",
--       fg         = "#3C6746",
--
--       number     = opts.background == "warm" and "#9ABDA0" or "#99C7AC",
--
--       white      = "#D9D3CE",
--       gray       = "#444E52",
--       green      = "#73A08D",
--       green0     = "#6FA791",
--       green1     = "#5E800E",
--       yellow     = "#BF7021",
--       yellow0    = "#C78500",
--       orange     = "#BF442B",
--       blue       = "#4F8FA1",
--       blue0      = "#4F6980",
--       lightblue  = "#0E747B",
--       lightgreen = "#107B6B",
--       pink       = "#913069",
--       cyan       = "#07790B",
--       cyan0      = "#00996D",
--       red        = "#971015",
--       red0       = "#FA5056",
--       red1       = "#E89396",
--
--       cursorline = opts.background == "warm" and "#BEDBB8" or "#BDDBC9",
--       separator  = "#9FB4A4",
--
--       statusbar  = opts.background == "warm" and "#C3DBBD" or "#C4DECE",
--       status_sep = opts.background == "warm" and "#AFCBA9" or "#B1D3BE",
--
--       bg_float   = opts.background == "warm" and "#B0CCAD" or "#B4D0BF",
--
--       bg_visual  = opts.background == "warm" and "#B4E1B2" or "#B3E0C5",
--
--       diff = {
--         add = "#9CC9B0",
--         change = "#BBD3B6",
--         delete = "#D3B6B6",
--       },
--
--       git = {
--         add = "#6EBB30",
--         change = "#218BE8",
--         delete = "#FF0008",
--       },
--
--       none = "NONE",
--     }
--   end
-- end
--
-- return M
