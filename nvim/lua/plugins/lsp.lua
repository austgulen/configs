return {
  {
    "neovim/nvim-lspconfig",
    opts = {
      servers = {
        -- BasedPyright: Global defaults (overridden by pyproject.toml)
        basedpyright = {
          settings = {
            basedpyright = { -- Root key for BasedPyright (not "python")
              analysis = {
                -- ---- Speed & Scope ─────
                diagnosticMode = "openFilesOnly", -- Only current file (fast)
                typeCheckingMode = "basic", -- Basic errors only
                useLibraryCodeForTypes = false, -- Skip library analysis
                autoSearchPaths = false, -- No auto-dir scanning
                extraPaths = { "src" }, -- Default source dirs (adjust globally)

                -- ---- Features ─────
                disableOrganizeImports = true, -- Use Ruff for imports

                -- ---- Diagnostic Overrides (global fallbacks) ─────
                diagnosticSeverityOverrides = {
                  reportUnknownVariableType = "none",
                  reportUnknownParameterType = "none",
                  reportMissingTypeAnnotation = "none",
                  reportGeneralTypeIssues = "none",
                  reportMissingTypeStubs = "none",
                  reportUnusedImport = "info", -- Keep as warning
                },
              },
              configurationPreference = "preferConfigFiles", -- Prioritize pyproject.toml
            },
          },
        },

        -- Ruff: Keep your fast linting (minimal rules)
        ruff = {
          init_options = {
            settings = {
              lint = {
                select = { "E4", "E7", "E9" },
                ignore = { "F" }, -- Let BasedPyright handle pyflakes
              },
              configurationPreference = "editorFirst", -- Use your global ruff.toml
            },
          },
        },
      },
    },
  },

  -- Ensure Ruff formatting uses global config (avoids local overrides)
  {
    "stevearc/conform.nvim",
    opts = {
      formatters = {
        ruff_format = {
          extra_args = { "--config", "/Users/a.austgulen/.config/ruff/ruff.toml" },
        },
      },
    },
  },
}
