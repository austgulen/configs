-- lua/config/conform.lua
return {
  "stevearc/conform.nvim",
  opts = {
    formatters_by_ft = {
      -- Markdown & MDX
      markdown = { "prettierd" },
      mdx = { "prettierd" },

      -- TeX
      tex = { "latexindent" },

      -- Programming languages
      lua = { "stylua" },
      python = {
        "ruff_format",
        "ruff_organize_imports",
        -- ruff_fix
      },
      javascript = { "prettierd" },
      typescript = { "prettierd" },
      go = { "gofmt" },
      rust = { "rustfmt" },
      sh = { "shfmt" },

      -- add more as desired…
    },

    -- ✨ Optional extras:
    -- formatter_opts = {
    --   prettier = { cli_args = { "--single-quote", "--prose-wrap=never" } },
    --   black    = { cli_args = { "--fast" } },
    -- },
    -- filter = function(ft, buf)
    --   -- only run prettier if package.json exists
    --   return ft ~= "javascript" or vim.loop.fs_stat("./package.json")
    -- end,
    -- concurrent = true,
  },
}
