return {
  "folke/which-key.nvim",
  opts = {
    spec = {
      -- text and note taking
      { "<leader>T", group = "text+" },
      { "<leader>Tm" },

      -- Diagnostics and DAP group
      { "<leader>d", group = "diagnostiiics" },
      {
        "<leader>dT",
        function()
          require("config.diagnostics").toggle()
        end,
        desc = "Toggle Warnings",
      },
    },
  },
}
