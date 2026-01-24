local M = {}

-- Diagnostic level: 0 = Errors only, 1 = +Warnings, 2 = +Hints
local diag_level = 0

local function get_min_severity()
  if diag_level == 0 then
    return vim.diagnostic.severity.ERROR
  elseif diag_level == 1 then
    return vim.diagnostic.severity.WARN
  else
    return vim.diagnostic.severity.HINT
  end
end

local function set_diag()
  local min_sev = get_min_severity()

  vim.diagnostic.config({
    severity_sort = true,
    underline = true,
    update_in_insert = false,
    signs = true,
    virtual_text = {
      prefix = "● ",
      source = "if_many",
      severity = { min = min_sev },
      format = function(d)
        local code = d.code or (d.user_data and d.user_data.lsp and d.user_data.lsp.code)
        return code and (d.message .. " [" .. code .. "]") or d.message
      end,
    },
    float = {
      border = "rounded",
      source = true,
      severity = { min = vim.diagnostic.severity.HINT },
    },
  })
end

function M.toggle()
  diag_level = (diag_level + 1) % 3
  set_diag()

  local status = ({
    [0] = "Errors only",
    [1] = "Errors + Warnings",
    [2] = "Errors + Warnings + Hints",
  })[diag_level]

  vim.notify("Diagnostics: " .. status)
end

-- Default: Errors only
set_diag()

return M
