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
    tag = "*",
    lazy = false,
    priority = 1000,
    -- configuration is optional!
    opts = {
      -- your settings here
    },
  },
  {
    "alljokecake/naysayer-theme.nvim",
    name = "naysayer", -- Optional, but good practice
  },

  -- set the final colorscheme
  {
    "LazyVim/LazyVim",
    opts = {
      -- colorscheme = "carbonfox",
      colorscheme = "bluloco",
    },
  },
}
