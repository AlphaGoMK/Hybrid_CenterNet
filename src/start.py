# coding:utf-8

import _init_paths
# 引入opt参数

from flask import Flask
from datetime import datetime, timedelta
import sys, uuid, os, werkzeug, shutil, cv2
import os.path as op
from flask import Flask, url_for, request, render_template, session, redirect
from opts import opts
from demo import det_eval

# 这里应该将static_url_path设为空，否则html中的每个资源连接都要以static开头才行，但是static_folder不要动，当来一个请求url时，会到static_folder下找静态文件，但是也会匹配static_url_path开头
app = Flask(__name__, static_url_path='')  # ,static_folder='',

upload_path = "./static/upload"
result_path = "./static/results"
heatmap_path = "./static/heatmap"

getfilename = lambda x: op.splitext(op.split(x)[-1])[0]



# html中只有h264编码的mp4视频可以播放
# 查看编码格式ffprobe file.mp4 -show_streams -select_streams v -print_format json
# ffmpeg的libcudart问题
# export PATH=$PATH:/usr/local/cuda-8.0/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64
# export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-8.0/lib64

def convert_mp4_h264(inpath):
    # cmdstr = "ffmpeg -i %s -vcodec libx264 -f mp4 %s" % (inpath, outpath)
    outpath = inpath.replace('.mp4', '_h264.mp4')
    cmdstr = 'ffmpeg -i %s -vcodec h264 %s' % (inpath, outpath)
    print(cmdstr)
    retn = os.system(cmdstr)
    if not retn:  # right
        if op.exists(inpath): os.remove(inpath)
        return outpath
    if op.exists(outpath): os.remove(outpath)
    return inpath


def clean_videos():
    if op.exists(upload_path): shutil.rmtree(upload_path)
    if op.exists(result_path):  shutil.rmtree(result_path)
    if op.exists(heatmap_path): shutil.rmtree(heatmap_path)


def random_filename(filename):
    ext = os.path.splitext(filename)[1]
    new_filename = uuid.uuid4().hex + ext
    return new_filename


# 主页面---------------------------------------------------------------------------
@app.route('/', methods=['GET', 'POST'])  # 主页面
def main_page():
    app.logger.debug(session)

    uploadresult_path = outpath = ''
    cam_li = []
    isImage = True

    tep = session.get('is_image')
    if tep is not None:
        isImage = tep

    tep = session.get('file_after')
    if tep is not None:
        outpath = tep

    tep = session.get('file_fps')
    if tep is not None:
        uploadresult_path = tep

    tep = session.get('cam_li')
    if tep is not None:
        cam_li = tep

    if isImage:
        return render_template('new_index.html', image_name_ori=getfilename(uploadresult_path),
                               image_name_res=getfilename(outpath), img_hm=cam_li,
                               is_image=isImage)
    else:
        return render_template('new_index.html', video_name_ori=getfilename(uploadresult_path),
                               video_name_res=getfilename(outpath),
                               is_image=isImage)


opt = opts().init()
opt.keep_res = True

opt.exp_id = 'coco_test'
opt.vis_cam = True
opt.nms = True

@app.route('/upload_file', methods=['POST'])  # file
def uploadfile():
    # 删除上一次的video
    tep = session.get('file_after')

    if tep is not None:
        if op.exists(tep): os.remove(tep)
        print("remove ", tep)
        session.pop("file_after")
    # 删除上一次的video
    tep = session.get('file_fps')

    if tep is not None:
        if op.exists(tep): os.remove(tep)
        print("remove ", tep)
        session.pop("file_fps")

    files = request.files.get('file')
    nms_thres = request.form.get('nms_thres', type=float, default=1)/10 # range只能获得整数，转为小数
    hyb_model = request.form.get('hyb_model', type=bool, default=False) # 没有选择checkbox时，没有hyb属性，选择默认值False
    print('hyb_model', hyb_model)
    print('nms_thres', nms_thres)
    opt.vis_thresh = nms_thres

    if not hyb_model:
        opt.load_model = '../exp/ctdet/v0_03coco_full/model_last.pth'
        opt.arch = 'dlav0_34'
    else:
        opt.load_model = '../exp/ctdet/camsplit_80_35_full/model_last.pth'
        opt.arch = 'dlav0camsplit_34'

    if files is not None:
        filename = random_filename(files.filename)

        uploadresult_path = op.join(upload_path, filename)
        files.save(uploadresult_path)

        if uploadresult_path.endswith('.jpg'):
            session['is_image'] = True
        else:
            session['is_image'] = False

        outpath, cam_li = det_eval(opt, uploadresult_path)

        if not uploadresult_path.endswith('.jpg'): # 视频必须转为h264编码才可以播放
            uploadresult_path = convert_mp4_h264(uploadresult_path)
            outpath = convert_mp4_h264(outpath)

        os.system('cp %s %s' % (outpath, result_path))

        session['file_fps'] = uploadresult_path
        session['file_after'] = op.join(result_path, getfilename(outpath))
        session['cam_li'] = cam_li

    return redirect(url_for("main_page"))


# ----------------------------------------------------------------------------------------
# 下面错误处理

@app.errorhandler(404)
def not_found(e):
    print("not found:", request.url)
    return '404 not found <h1>' + request.url + '</h1>'


# ---------------------------------------------------------------------------------------

if __name__ == '__main__':
    clean_videos()

    os.makedirs(upload_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(heatmap_path, exist_ok=True)

    app.send_file_max_age_default = timedelta(seconds=1)
    app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 30M
    app.secret_key = os.urandom(24)
    app.run(host='0.0.0.0', port=9766)
