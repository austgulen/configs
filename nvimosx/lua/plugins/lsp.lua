return {
  "neovim/nvim-lspconfig",
  opts = {
    servers = {
      ruff = {
        -- https://docs.astral.sh/ruff/editors/settings/
        init_options = {
          settings = {
            -- https://docs.astral.sh/ruff/configuration/
            configuration = "~/.config/ruff/ruff.toml",
            -- logLevel = "debug",
          },
        },
      },
      -- https://docs.basedpyright.com/latest/
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

      -- basedpyright = {
      --   settings = {
      --     basedpyright = {
      --       analysis = {
      --         typeCheckingMode = "basic", -- Less strict, fewer diagnostics
      --         diagnosticMode = "openFilesOnly", -- Faster, less verbose
      --         -- useLibraryCodeForTypes = false, -- Faster for simple projects
      --         autoSearchPaths = true, -- Adjust based on project size
      --         disableOrganizeImports = true, -- Let Ruff handle imports
      --         diagnosticSeverityOverrides = {
      --           reportUnusedImport = "warning", -- Less intrusive
      --           -- reportGeneralTypeIssues = "warning", -- Downgrade type issues
      --           -- Remove annoying warnings:
      --           reportMissingTypeAnnotation = "none",
      --           reportMissingParameterType = "none",
      --           reportUnknownVariableType = "none",
      --           reportUnknownParameterType = "hint",
      --           reportGeneralTypeIssues = "none", -- New: Skip general type issues
      --           reportMissingTypeStubs = "none", -- New: Skip missing stub warnings
      --           reportOptionalMemberAccess = "none",
      --         },
      --       },
      --     },
      --   },
      -- },
    },
  },
}
