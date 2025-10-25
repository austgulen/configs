return {
  -- import themes
  {
    "Everblush/nvim",
    name = "Everblush",
  },
  {
    "jpwol/thorn.nvim",
    lazy = false,
    priority = 1000,
    opts = {},
  },
  {
    "rebelot/kanagawa.nvim",
    lazy = false,
    priority = 1000,
    opts = {},
    -- config = function()
    --   require("kanagawa").setup({ flavor = " wave" })
    --   vim.cmd.colorscheme("kanagawa")
    -- end,
  },
  {
    "craftzdog/solarized-osaka.nvim",
    lazy = false,
    priority = 1000,
    opts = {},
  },
  {
    "ribru17/bamboo.nvim",
    lazy = false,
    priority = 1000,
    config = function()
      -- require("bamboo").setup({
      -- optional configuration here
      -- })
      -- require("bamboo").load()
    end,
  },
  {
    "uloco/bluloco.nvim",
    lazy = false,
    priority = 1000,
    dependencies = { "rktjmp/lush.nvim" },
    config = function()
      require("bluloco").setup({
        style = "dark", -- "auto" | "dark" | "light"
        -- transparent = false,
        italics = true,
        terminal = vim.fn.has("gui_running") == 1, -- bluoco colors are enabled in gui terminals per default.
        guicursor = true,
        rainbow_headings = false, -- if you want different colored headings for each heading level
      })
      -- your optional config goes here, see below.
    end,
  },
  { "bluz71/vim-nightfly-colors", name = "nightfly", lazy = false, priority = 1000 },
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
    "folksoftware/nvim",
    name = "folk",
    priority = 1000,
    -- config = function()
    --   require("folk").setup({ flavour = "abraxas" })
    --   vim.cmd.colorscheme("folk-abraxas")
    -- end,
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
  { "bluz71/vim-moonfly-colors", name = "moonfly", lazy = false, priority = 1000 },
  -- set the final colorscheme
  -- Load the plugin
  { "craftzdog/solarized-osaka.nvim", lazy = false, priority = 1000 },

  -- -- Tell LazyVim to activate it
  -- {
  --   "LazyVim/LazyVim",
  --   opts = { colorscheme = "solarized-osaka" },
  -- },

  -- Configure the theme (all options from its docs)
  {
    "craftzdog/solarized-osaka.nvim",
    opts = {
      -- transparent = true,
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
      transparent = true, -- Set to true for no background
      dim_inactive = false, -- Dim inactive windows
      lualine_bols = true,

      on_highlights = function(hl, c) -- Optional: Custom highlight overrides
        hl["@field"] = { fg = c.green }
        hl["@property"] = { fg = c.blue }
      end,
    },
  },
  {
    "LazyVim/LazyVim",
    opts = {
      -- colorscheme = "matteblack",
      -- colorscheme = "carbonfox",
      -- colorscheme = "bluloco",
      -- colorscheme = "solarized-osaka",
      colorscheme = "cyberdream",
    },
  },
}
