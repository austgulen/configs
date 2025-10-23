return {
  "neovim/nvim-lspconfig",
  opts = {
    servers = {
      ruff = {
        -- https://docs.astral.sh/ruff/editors/settings/
        init_options = {
          settings = {
            lineLength = 100,
            logLevel = "info",
            lint = {
              enable = true,
            },
            format = {
              preview = true,
            },
          },
        },
      },

      basedpyright = {
        settings = {
          basedpyright = {
            analysis = {
              --diagnosticMode = "openFilesOnly",
              --typeCheckingMode = "basic",
            },
          },
        },
      },
    },
  },
}
