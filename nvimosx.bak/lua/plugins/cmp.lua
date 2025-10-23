return {
  "hrsh7th/nvim-cmp",
  dependencies = {
    "L3MON4D3/LuaSnip", -- Snippet engine
    "rafamadriz/friendly-snippets", -- Collection of snippets
    "saadparwaiz1/cmp_luasnip", -- LuaSnip source for nvim-cmp
    "hrsh7th/cmp-nvim-lsp", -- LSP completion source
    "hrsh7th/cmp-buffer", -- Buffer words completion
    "hrsh7th/cmp-path", -- File system paths
    "hrsh7th/cmp-cmdline", -- Vim command-line completion
    "onsails/lspkind-nvim", -- Adds VSCode-like icons
    "hrsh7th/cmp-nvim-lsp-signature-help",
  },
  opts = function(_, opts)
    local cmp = require("cmp")
    local luasnip = require("luasnip")
    local lspkind = require("lspkind")

    -- Load VSCode-style snippets from friendly-snippets
    require("luasnip.loaders.from_vscode").lazy_load()

    -- Use VSCode-like pictograms with text
    opts.formatting = {
      format = lspkind.cmp_format({
        mode = "symbol_text", -- show symbol + text
        maxwidth = 50, -- prevent the popup from growing too wide
        ellipsis_char = "...",
      }),
    }

    -- Recommended Vim completion settings
    vim.opt.completeopt = { "menu", "menuone", "noselect", "preview" }

    -- Snippet expansion function
    opts.snippet = {
      expand = function(args)
        luasnip.lsp_expand(args.body)
      end,
    }

    -- Define the order & priority of completion sources
    opts.sources = cmp.config.sources({
      { name = "nvim_lsp", priority = 1000 },
      { name = "luasnip", priority = 750 },
      { name = "buffer", priority = 500 },
      { name = "path", priority = 250 },
    })

    -- Key mappings for insert & select modes
    opts.mapping = cmp.mapping.preset.insert({
      ["<C-Space>"] = cmp.mapping.complete(), -- manually trigger completion menu
      ["<CR>"] = cmp.mapping.confirm({ select = false }), -- confirm selection, do not auto-select
      ["<C-e>"] = cmp.mapping.abort(), -- close completion menu
      ["<C-d>"] = cmp.mapping.scroll_docs(-4), -- scroll docs up
      ["<C-f>"] = cmp.mapping.scroll_docs(4), -- scroll docs down
      ["<Tab>"] = cmp.mapping(function(fallback) -- smart Tab: next item or snippet jump
        if cmp.visible() then
          cmp.select_next_item()
        elseif luasnip.expand_or_jumpable() then
          luasnip.expand_or_jump()
        else
          fallback()
        end
      end, { "i", "s" }),
      ["<S-Tab>"] = cmp.mapping(function(fallback) -- smart Shift-Tab: prev item or snippet back
        if cmp.visible() then
          cmp.select_prev_item()
        elseif luasnip.jumpable(-1) then
          luasnip.jump(-1)
        else
          fallback()
        end
      end, { "i", "s" }),
    })

    -- Experimental ghost text preview (inline suggestion)
    opts.experimental = {
      ghost_text = true,
    }
  end,
}
