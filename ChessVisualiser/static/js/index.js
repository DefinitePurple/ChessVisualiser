$(document).ready(function() {
    let current = 0;
    moves.unshift('start');

    let cfg = {
        draggable: false,
        dropOffBoard: 'snap',
        position: moves[0]
    };
    let board = ChessBoard('board', cfg);

    $('#reset').click(function() {
        current = 0;
        cfg.position = 'start';
        board = ChessBoard('board', cfg);
    });

    $('#previous').click(function() {
        if (current !== 0) {
            let curMove = moves[current].split('-');
            curMove = curMove[1] + '-' + curMove[0];
            board.move(curMove);
            current = current - 1;
        }
    });

    $('#next').click(function() {
        if (current < moves.length-1) {
            current = current + 1;
            board.move(moves[current]);
        }
    });

    $('#last').click(function() {
        moves.forEach(function(move) {
            current = current + 1;
            board.move(move);
        });
    });

    const moveDetails = getMoveDetails(moves, startingPosition);
    displayDetails(moveDetails);
});

function displayDetails(details) {
    details.forEach(function(detail){
        if(detail.colour === 'white')
           createRow(detail.moveCount);

        let el = 'div#row-'+ detail.moveCount;
        console.log($(el).find('span#' + detail.colour));
        $(el).find('span#' + detail.colour).html(detail.pgn);
    });
}

function createRow(count) {
    const row = document.createElement('div');
    row.id = 'row-' + count;
    row.classList.add('row');
    const number = document.createElement('span');
    number.id = 'move';
    number.classList.add('col-1');
    number.innerText = count + '.';
    const white = document.createElement('span');
    white.id = 'white';
    white.classList.add('col-4');
    const black = document.createElement('span');
    black.id = 'black';
    black.classList.add('col-3');

    row.appendChild(number);
    row.appendChild(white);
    row.appendChild(black);
    $('#vertical-moves').append(row);
}

function getMoveDetails(moves, boardState) {
    let list = [];
    let count = 0;
    for (let i = 1; i < moves.length; i++) {
        count = (i % 2 === 0 ) ? count : count + 1;
        list.push(new MoveDetails(moves[i], count, boardState));

        let move = moves[i].split('-');
        move = move.reduce((acc, cur) => acc + '' + cur);
        move = move.split('');

        let piece = boardState[move[1]-1][mapFile(move[0])];
        boardState[move[1] - 1][mapFile(move[0])] = 0;
        boardState[move[3] - 1][mapFile(move[2])] = piece

    }
    console.log(list);
    return list;
}

class MoveDetails {
    constructor(squareToSquare, count, boardState) {
        this.squareToSquare = squareToSquare;
        this.colour = 'white';
        this.moveCount = count;
        this.getPGNMove(squareToSquare, boardState);
    }

    getPGNMove(move, boardState) {
        move = move.split('-');
        move = move.reduce((acc, cur) => acc + '' + cur);
        move = move.split('');

        let piece = boardState[move[1]-1][mapFile(move[0])];
        let otherPiece = boardState[move[3] - 1][mapFile(move[2])];

        if(piece === piece.toLowerCase())
            this.colour = 'black';

        if (piece.toLowerCase() === 'p' && move[0] !== move[2]) {
            this.pgn = move[1] + 'x' + move[2] + move[3];
        } else if (piece.toLowerCase() === 'p') {
            this.pgn = move[2] + move[3];
        } else if (otherPiece !== 0) {
            this.pgn = piece.toUpperCase() + 'x'  + move[2] + move[3];
        } else {
            this.pgn = piece.toUpperCase() + move[2] + move[3];
        }
    }
}

function mapFile(letter) {
    return letter.charCodeAt(0) - 97;
}


