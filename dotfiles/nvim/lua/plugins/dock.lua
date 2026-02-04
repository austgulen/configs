return {
  "snacks.nvim",
  opts = {
    dashboard = {
      preset = {
        pick = function(cmd, opts)
          return LazyVim.pick(cmd, opts)()
        end,
        header = [[
                                                        ,,                    
              `7MN.   `7MF'                             db                    
`\\.            MMN.    M                                                     
   `\\:.        M YMb   M  .gP"Ya   ,pW"Wq.`7M'   `MF'`7MM  `7MMpMMMb.pMMMb.  
      `\\.      M  `MN. M ,M'   Yb 6W'   `Wb VA   ,V    MM    MM    MM    MM  
     ,;//'      M   `MM.M 8M"""""" 8M     M8  VA ,V     MM    MM    MM    MM  
  ,;//'         M     YMM YM.    , YA.   ,A9   VVV      MM    MM    MM    MM  
,//'          .JML.    YM  `Mbmmd'  `Ybmd9'     W     .JMML..JMML  JMML  JMML.
]],
        -- vim.api.nvim_set_hl(0, "SnacksDashboardHeader", { fg = "#ff0086" }),
        -- stylua: ignore
        ---@type snacks.dashboard.Item[]
        keys = {
          { icon = " ", key = "f", desc = "Find File", action = ":lua Snacks.dashboard.pick('files')" },
          { icon = " ", key = "n", desc = "New File", action = ":ene | startinsert" },
          { icon = " ", key = "g", desc = "Find Text", action = ":lua Snacks.dashboard.pick('live_grep')" },
          { icon = " ", key = "r", desc = "Recent Files", action = ":lua Snacks.dashboard.pick('oldfiles')" },
          { icon = " ", key = "c", desc = "Config", action = ":lua Snacks.dashboard.pick('files', {cwd = vim.fn.stdpath('config')})" },
          { icon = " ", key = "s", desc = "Restore Session", section = "session" },
          { icon = " ", key = "x", desc = "Lazy Extras", action = ":LazyExtras" },
          { icon = "󰒲 ", key = "l", desc = "Lazy", action = ":Lazy" },
          { icon = " ", key = "q", desc = "Quit", action = ":qa" },
        },
      },
    },
  },
}
