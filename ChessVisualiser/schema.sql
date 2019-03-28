-- run 'createdb chess'
-- create extension pgcrypto -- must be run at least once when the db is created
drop table Match;
drop table Users;

create table Users (
	id serial PRIMARY KEY,
	email character varying NOT NULL unique,
	username character varying NOT NULL unique ,
	password character varying NOT NULL
);

create table Match(
  id serial PRIMARY KEY,
  user_id serial REFERENCES Users(id),
  white character varying,
  black character varying,
  score character varying,
  moves character varying,
  file character varying,
  url character varying,
  date character varying
);


CREATE OR REPLACE FUNCTION login_user (IN p_username character varying, IN p_password character varying)
returns table (
  password boolean
) AS $$
BEGIN
  return query
    select u.password = crypt(p_password, u.password) as password from Users u where LOWER(u.username) = LOWER(p_username);
END; $$
language 'plpgsql';

CREATE OR REPLACE PROCEDURE create_user (
  IN p_email character varying,
  IN p_username character varying,
  IN p_password character varying
) AS $$
BEGIN
  insert into Users(email, username, password) values (p_email, p_username, crypt(p_password, gen_salt('md5')));
END; $$
language 'plpgsql';

-- CALL create_user ('www@www.com', 'definitepurple', 'password'); -- test create_user procedure