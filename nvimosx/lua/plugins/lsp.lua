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
