from flask import Flask, render_template, session, request
from flask_socketio import SocketIO, emit, join_room
from persuasion_bot import PersuasiveBot

# Init the server
app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app)
acssModel = PersuasiveBot()
clients = []


@app.route("/persuasionchatbot")
def root():
    """Send HTML from the server."""
    return render_template("usc_index.html")


@socketio.on("joined")
def joined(message):
    """Sent by clients when they enter a room.
    A status message is broadcast to all people in the room."""
    # Add client to client list
    clients.append(request.sid)
    print(request.sid)

    room = session.get("room")
    join_room(room)
    acssModel.reload()
    out_text = acssModel.chat("", request.sid)
    emit("start_sys_message", {"data": out_text}, room=request.sid)

    # emit to the first client that joined the room
    # emit('status', {'msg': session.get('name') + ' has entered the room.'}, room=clients[0])


@socketio.on("user_sent_message")
def user_sent_message(message):
    """
    Called when the user sends a message.
    """

    # This renders the user message we have received on screen
    emit("render_usr_message", message, room=request.sid)

    # Extract a string of the users message
    input_text = message["data"]
    out_text = acssModel.chat(input_text, request.sid)

    # Render our response
    emit("render_sys_message", {"data": out_text}, room=request.sid)

    # emit('render_da', {"sys_da":sys_da_output, "sys_se": sys_se_output, "usr_da": usr_da_output, "usr_se": usr_se_output})


if __name__ == "__main__":
    """Run the app."""
    socketio.run(app, host="0.0.0.0", port=7090, use_reloader=False)
