export ZSH="$HOME/.oh-my-zsh"

if [[ -r "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh" ]]; then
  source "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh"
fi

# --- Oh My Zsh Setup ---
ZSH_THEME="powerlevel10k/powerlevel10k"
ZSH_AUTOSUGGEST_HIGHLIGHT_STYLE="fg=#586e75" # Solarized gray
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

eval "$(zoxide init --cmd cd zsh)"

# History settings
HISTFILE=~/.histfile
HISTSIZE=10000
SAVEHIST=10000
setopt hist_ignore_dups   # Ignore duplicate commands
setopt share_history      # Share history across sessions
setopt inc_append_history # Append commands immediately

# Aliases
# alias ls='ls --color=auto'
alias vim='nvim'
# alias ll='ls -lh --color=auto'
# alias la='ls -lha --color=auto'
alias ls='eza --icons --group-directories-first'
alias ll='eza -lh --icons --git --group-directories-first'
alias la='eza -lha --icons --git --group-directories-first'
# alias lt='eza --tree --level=2 --icons' # Quick tree view
alias update='sudo pacman -Syu'
alias install='sudo pacman -S'
alias rm='rm -i'
alias cp='cp -i'
alias mv='mv -i'
alias src='source ~/.zshrc'
alias ezsh='nvim ~/.zshrc'
alias rgi='rg --ignore-case --line-number --no-heading | fzf --ansi --preview "bat --style=numbers --color=always --line-range :500 {1}"'
lt() {
  local depth=${1:-2} # Set depth to $1, or default to 2 if $1 is empty
  eza --tree --level="$depth" --icons --group-directories-first "${@:2}"
}
# --- Prompt Customization ---
# Powerlevel10k loads its config from ~/.p10k.zsh
[[ ! -f ~/.p10k.zsh ]] || source ~/.p10k.zsh

# --- Extra Quality of Life ---
# Better completion
autoload -Uz compinit
zstyle ':completion:*' menu select
zstyle ':completion:*' matcher-list 'm:{a-z}={A-Z}' 'r:|[._-]=* r:|=*'
compinit

autoload -Uz edit-command-line
zle -N edit-command-line
bindkey '^X^E' edit-command-line


zstyle ':completion:*' menu select
zstyle ':completion:*' matcher-list 'm:{a-z}={A-Z}' 'r:|[._-]=* r:|=*'

bindkey '^[[A' history-substring-search-up
bindkey '^[[B' history-substring-search-down


fzf_history_widget() {
  BUFFER=$(history -n 1 | tac | fzf --tac --no-sort --ansi --height 40% --reverse --query="$LBUFFER" --preview 'echo {}' )
  CURSOR=$#BUFFER
  zle reset-prompt
}
zle -N fzf_history_widget
bindkey '^R' fzf_history_widget
export PATH="$HOME/.cargo/bin:$PATH"

# >>> juliaup initialize >>>

# !! Contents within this block are managed by juliaup !!

path=('/Users/kaust/.juliaup/bin' $path)
export PATH

# <<< juliaup initialize <<<
