# --- Powerlevel10k Instant Prompt ---
if [[ -r "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh" ]]; then
  source "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh"
fi

# --- Oh My Zsh Setup ---
export ZSH="$HOME/.oh-my-zsh"
ZSH_THEME="powerlevel10k/powerlevel10k"

plugins=(
  git
  zoxide
  zsh-autosuggestions
  zsh-syntax-highlighting
  # Add more plugins here: docker npm fzf
  fzf-tab
  zsh-completions
  zsh-history-substring-search
)

source $ZSH/oh-my-zsh.sh

# --- Custom Configuration ---

# Zoxide smart cd
eval "$(zoxide init --cmd cd zsh)"

# History settings
HISTFILE=~/.histfile
HISTSIZE=10000
SAVEHIST=10000
setopt hist_ignore_dups   # Ignore duplicate commands
setopt share_history      # Share history across sessions
setopt inc_append_history # Append commands immediately

# Aliases
alias ls='ls --color=auto'
alias vim='nvim'
alias ll='ls -lh --color=auto'
alias la='ls -lha --color=auto'
alias update='sudo pacman -Syu'
alias install='sudo pacman -S'
alias rm='rm -i'
alias cp='cp -i'
alias mv='mv -i'
alias src='source ~/.zshrc'
alias ezsh='nvim ~/.zshrc'
alias rgi='rg --ignore-case --line-number --no-heading | fzf --ansi --preview "bat --style=numbers --color=always --line-range :500 {1}"'

# --- Prompt Customization ---
# Powerlevel10k loads its config from ~/.p10k.zsh
[[ ! -f ~/.p10k.zsh ]] || source ~/.p10k.zsh

# --- Extra Quality of Life ---
# Better completion
autoload -Uz compinit
zstyle ':completion:*' menu select
zstyle ':completion:*' matcher-list 'm:{a-z}={A-Z}' 'r:|[._-]=* r:|=*'
compinit

# Less typing: edit command line in $EDITOR
autoload -Uz edit-command-line
zle -N edit-command-line
bindkey '^X^E' edit-command-line


# --- fzf-tab configuration ---
# Use fzf for completion menus
zstyle ':completion:*' menu select
zstyle ':completion:*' matcher-list 'm:{a-z}={A-Z}' 'r:|[._-]=* r:|=*'

# --- History substring search ---
bindkey '^[[A' history-substring-search-up
bindkey '^[[B' history-substring-search-down


# --- Fuzzy history search with fzf ---
# Ctrl+R will open an interactive fzf menu over shell history.
fzf_history_widget() {
  BUFFER=$(history -n 1 | tac | fzf --tac --no-sort --ansi --height 40% --reverse --query="$LBUFFER" --preview 'echo {}' )
  CURSOR=$#BUFFER
  zle reset-prompt
}
zle -N fzf_history_widget
bindkey '^R' fzf_history_widget

